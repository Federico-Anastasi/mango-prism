#include "server-slot-file.h"
#include "server-common.h"

#include "log.h"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr uint32_t SLOT_FILE_MAGIC   = 0x7073736du; // "mssp"
static constexpr uint32_t SLOT_FILE_VERSION = 2; // 2: turn start (turn_n)
static constexpr size_t   SLOT_FILE_FOOTER  = 8 + 8 + 8 + 4 + 4;

// Thresholds, in prompt tokens to reread (hundreds of tokens per second on the home GPUs).
static constexpr size_t DISK_MIN_GAIN = 256;          // a file is loaded if it reuses at least this many more tokens than the slot
static constexpr size_t DISK_MIN_SAVE = 1024;         // a state is saved if rebuilding it would cost at least this much
static constexpr size_t DISK_MIN_BASE = 1024;         // a shorter common prefix is not worth a file
static constexpr size_t DISK_MIN_FREE = 2ull << 30;   // at least 2 GiB always stay free on the disk

static uint64_t hash_bytes(const void * p, size_t n) {
    return XXH64(p, n, 0);
}

static uint64_t tokens_hash(const server_tokens & t) {
    const std::vector<char> v = t.serialize();
    return hash_bytes(v.data(), v.size());
}

static std::string base_file_name(uint64_t hash) {
    char name[40];
    snprintf(name, sizeof(name), "base-%016llx.slot", (unsigned long long) hash);
    return name;
}

// The base name depends only on its tokens (text only: system prompt and tools).
// (tokens are read one by one: get_tokens() rejects prompts with images)
static bool base_hash(const server_tokens & prompt, size_t base_n, uint64_t & h) {
    if (base_n > prompt.size()) {
        return false;
    }
    llama_tokens t(base_n);
    for (size_t i = 0; i < base_n; ++i) {
        t[i] = prompt[i];
        if (t[i] == LLAMA_TOKEN_NULL) {
            return false;
        }
    }
    h = hash_bytes(t.data(), base_n * sizeof(llama_token));
    return true;
}

size_t slot_disk_base_length(const llama_vocab * vocab, const std::string & base_text, const server_tokens & prompt) {
    if (base_text.empty()) {
        return 0;
    }
    const llama_tokens base = common_tokenize(vocab, base_text, true, true);
    size_t n = 0;
    while (n < base.size() && n < prompt.size() && base[n] == prompt[n] && prompt[n] != LLAMA_TOKEN_NULL) {
        ++n;
    }
    return n;
}

static int64_t file_time_s(const fs::file_time_type & t) {
    return std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
}

static int64_t now_s() {
    return file_time_s(fs::file_time_type::clock::now());
}

uint64_t slot_file_model_fingerprint(const common_params & params) {
    XXH64_state_t h;
    XXH64_reset(&h, 0);
    auto add_file = [&](const std::string & path) {
        const uint64_t size = fs::file_size(path);
        XXH64_update(&h, &size, sizeof(size));
        std::ifstream f(path, std::ios::binary);
        const uint64_t chunk = std::min<uint64_t>(size, 4ull << 20);
        std::vector<char> buf(chunk);
        f.read(buf.data(), chunk);
        XXH64_update(&h, buf.data(), chunk);
        f.seekg(size - chunk);
        f.read(buf.data(), chunk);
        if (!f) {
            throw std::runtime_error("cannot read " + path + " for the state fingerprint");
        }
        XXH64_update(&h, buf.data(), chunk);
    };
    add_file(params.model.path);
    if (!params.mmproj.path.empty()) {
        add_file(params.mmproj.path);
    }
    for (const auto & la : params.lora_adapters) {
        add_file(la.path);
        XXH64_update(&h, &la.scale, sizeof(la.scale));
    }
    const int32_t kv[2] = { (int32_t) params.cache_type_k, (int32_t) params.cache_type_v };
    XXH64_update(&h, kv, sizeof(kv));
    XXH64_update(&h, &SLOT_FILE_VERSION, sizeof(SLOT_FILE_VERSION));
    return XXH64_digest(&h);
}

// A part of the file: position, length, hash.
struct slot_file_part {
    uint64_t offset = 0;
    uint64_t size   = 0;
    uint64_t hash   = 0;
};

static json part_json(const slot_file_part & p) {
    return { { "offset", p.offset }, { "size", p.size }, { "hash", p.hash } };
}

static slot_file_part part_from_json(const json & j) {
    return { j.at("offset").get<uint64_t>(), j.at("size").get<uint64_t>(), j.at("hash").get<uint64_t>() };
}

struct slot_file_ckpt_meta {
    int64_t        n_tokens = 0;
    llama_pos      pos_min  = 0;
    llama_pos      pos_max  = 0;
    slot_file_part tgt, dft, spec;
};

struct slot_file_meta {
    uint64_t       model    = 0;
    bool           base     = false;
    int32_t        prompt_n = 0;
    int32_t        turn_n   = 0;
    slot_file_part tgt, dft, spec, tokens;
    std::vector<slot_file_ckpt_meta> ckpts;
};

// Writes a part and keeps its position and hash.
struct slot_file_writer {
    std::ofstream f;
    uint64_t      off = 0;

    slot_file_part write(const void * p, size_t n) {
        slot_file_part part = { off, n, hash_bytes(p, n) };
        if (n > 0) {
            f.write((const char *) p, n);
        }
        off += n;
        return part;
    }
};

static bool read_part(std::ifstream & f, const slot_file_part & p, std::vector<uint8_t> & out) {
    out.resize(p.size);
    if (p.size == 0) {
        return p.hash == hash_bytes(out.data(), 0);
    }
    f.seekg(p.offset);
    f.read((char *) out.data(), p.size);
    return f && hash_bytes(out.data(), out.size()) == p.hash;
}

// Footer and metadata, with the consistency checks that do not require reading the state.
static bool slot_file_read_meta(const std::string & path, slot_file_meta & meta, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "file not found";
        return false;
    }
    std::error_code ec;
    const uint64_t size = fs::file_size(path, ec);
    if (ec || size < SLOT_FILE_FOOTER) {
        err = "file truncated";
        return false;
    }
    uint64_t meta_off = 0, meta_size = 0, meta_hash = 0;
    uint32_t version = 0, magic = 0;
    f.seekg(size - SLOT_FILE_FOOTER);
    f.read((char *) &meta_off, 8).read((char *) &meta_size, 8).read((char *) &meta_hash, 8);
    f.read((char *) &version, 4).read((char *) &magic, 4);
    if (!f || magic != SLOT_FILE_MAGIC) {
        err = "not a state file (or truncated)";
        return false;
    }
    if (version != SLOT_FILE_VERSION) {
        err = "format version " + std::to_string(version) + ", expected " + std::to_string(SLOT_FILE_VERSION);
        return false;
    }
    if (meta_off + meta_size + SLOT_FILE_FOOTER != size) {
        err = "footer inconsistent with the file size";
        return false;
    }
    std::string m(meta_size, '\0');
    f.seekg(meta_off);
    f.read(m.data(), meta_size);
    if (!f || hash_bytes(m.data(), m.size()) != meta_hash) {
        err = "corrupted metadata";
        return false;
    }
    try {
        const json j = json::parse(m);
        meta.model    = j.at("model").get<uint64_t>();
        meta.base     = j.at("base").get<bool>();
        meta.prompt_n = j.at("prompt_n").get<int32_t>();
        meta.turn_n   = j.at("turn_n").get<int32_t>();
        meta.tgt      = part_from_json(j.at("tgt"));
        meta.dft      = part_from_json(j.at("dft"));
        meta.spec     = part_from_json(j.at("spec"));
        meta.tokens   = part_from_json(j.at("tokens"));
        uint64_t end  = meta.tgt.size + meta.dft.size + meta.spec.size;
        if (meta.tgt.offset != 0 || meta.dft.offset != meta.tgt.size || meta.spec.offset != meta.tgt.size + meta.dft.size ||
            meta.tgt.size == 0) {
            err = "state out of place in the metadata";
            return false;
        }
        for (const auto & c : j.at("checkpoints")) {
            slot_file_ckpt_meta cm;
            cm.n_tokens = c.at("n_tokens").get<int64_t>();
            cm.pos_min  = c.at("pos_min").get<llama_pos>();
            cm.pos_max  = c.at("pos_max").get<llama_pos>();
            cm.tgt      = part_from_json(c.at("tgt"));
            cm.dft      = part_from_json(c.at("dft"));
            cm.spec     = part_from_json(c.at("spec"));
            for (const auto * p : { &cm.tgt, &cm.dft, &cm.spec }) {
                if (p->offset != end) {
                    err = "checkpoint out of place in the metadata";
                    return false;
                }
                end += p->size;
            }
            meta.ckpts.push_back(cm);
        }
        if (meta.tokens.offset != end || meta.tokens.offset + meta.tokens.size != meta_off) {
            err = "the file parts do not match the metadata";
            return false;
        }
    } catch (const std::exception & e) {
        err = std::string("unreadable metadata: ") + e.what();
        return false;
    }
    return true;
}

static bool read_tokens(const std::string & path, const slot_file_part & p, bool has_mtmd, server_tokens & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> raw;
    if (!read_part(f, p, raw) || raw.size() % sizeof(llama_token) != 0) {
        err = "corrupted tokens";
        return false;
    }
    llama_tokens packed(raw.size() / sizeof(llama_token));
    memcpy(packed.data(), raw.data(), raw.size());
    try {
        out = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & e) {
        err = std::string("unreadable tokens: ") + e.what();
        return false;
    }
    return true;
}

// The most advanced checkpoint up to `limit` tokens, excluding `skip`.
static const common_prompt_checkpoint * last_upto(const server_prompt & p, int64_t limit, const common_prompt_checkpoint * skip = nullptr) {
    const common_prompt_checkpoint * best = nullptr;
    for (const auto & c : p.checkpoints) {
        if (&c != skip && !c.data_tgt.empty() && c.n_tokens <= limit && (!best || c.n_tokens > best->n_tokens)) {
            best = &c;
        }
    }
    return best;
}

bool slot_disk_protected(const server_prompt & p, const slot_disk_state & st, const common_prompt_checkpoint & c) {
    if (st.turn_n <= 0) {
        return false;
    }
    const common_prompt_checkpoint * a = last_upto(p, st.turn_n);
    return a && (&c == a || &c == last_upto(p, st.turn_n, a));
}

// The checkpoints that go into the file: the two at the start of the turn (the next turn renders the turn's
// answers without reasoning and diverges there; two, because the exact point depends on how the template closes
// the prompt), the one at the end of the last prompt and the last one (a step after one that diverges towards the
// end, for example a rewritten tool call). Those in between would only serve prompts that diverge earlier, which
// a saved conversation never sees again: with 32 checkpoints of tens of MB the file would grow by gigabytes.
static std::vector<const common_prompt_checkpoint *> file_ckpts(const server_prompt & p, size_t n_tok, const slot_disk_state & st, bool base) {
    std::vector<const common_prompt_checkpoint *> keep;
    if (base) {
        return keep;
    }
    auto add = [&](const common_prompt_checkpoint * c) {
        if (c && c->n_tokens <= (int64_t) n_tok && std::find(keep.begin(), keep.end(), c) == keep.end()) {
            keep.push_back(c);
        }
    };
    const common_prompt_checkpoint * turn = last_upto(p, st.turn_n);
    add(turn);
    add(turn ? last_upto(p, st.turn_n, turn) : nullptr);
    add(last_upto(p, st.prompt_n));
    add(last_upto(p, (int64_t) n_tok));
    std::sort(keep.begin(), keep.end(), [](auto * a, auto * b) { return a->n_tokens < b->n_tokens; });
    return keep;
}

static size_t state_size(const slot_file_view & s, size_t n_tok, const slot_disk_state & st, bool base) {
    size_t n = llama_state_seq_get_size_ext(s.ctx_tgt, s.id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (s.ctx_dft) {
        n += llama_state_seq_get_size_ext(s.ctx_dft, s.id, LLAMA_STATE_SEQ_FLAGS_NONE);
    }
    for (const auto * c : file_ckpts(*s.prompt, n_tok, st, base)) {
        n += c->size();
    }
    return n + n_tok * 8 + (1u << 20);
}

// Writes the current state of the slot's memory, which corresponds to the first `n_tok` cached tokens.
static slot_file_result save_n(const slot_file_view & s, const std::string & path, uint64_t model_fp, bool base,
                               const slot_disk_state & st, size_t n_tok) {
    const int64_t t0 = ggml_time_us();
    slot_file_result res;
    const std::string tmp = path + ".tmp";
    auto fail = [&](const std::string & why) {
        std::error_code ec;
        fs::remove(tmp, ec);
        res.error = why;
        return res;
    };

    server_tokens toks = s.prompt->tokens.clone();
    toks.keep_first(n_tok);
    std::vector<char> packed;
    try {
        packed = toks.serialize();
    } catch (const std::exception & e) {
        return fail(std::string("tokens cannot be saved: ") + e.what());
    }

    slot_file_writer w;
    w.f.open(tmp, std::ios::binary | std::ios::trunc);
    if (!w.f) {
        return fail("cannot open " + tmp);
    }
    std::vector<uint8_t> buf(llama_state_seq_get_size_ext(s.ctx_tgt, s.id, LLAMA_STATE_SEQ_FLAGS_NONE));
    if (buf.empty() || llama_state_seq_get_data_ext(s.ctx_tgt, buf.data(), buf.size(), s.id, LLAMA_STATE_SEQ_FLAGS_NONE) != buf.size()) {
        return fail("reading the model state failed");
    }
    const slot_file_part tgt = w.write(buf.data(), buf.size());
    buf.clear();
    if (s.ctx_dft) {
        buf.resize(llama_state_seq_get_size_ext(s.ctx_dft, s.id, LLAMA_STATE_SEQ_FLAGS_NONE));
        if (llama_state_seq_get_data_ext(s.ctx_dft, buf.data(), buf.size(), s.id, LLAMA_STATE_SEQ_FLAGS_NONE) != buf.size()) {
            buf.clear(); // the draft is optional: without it, it restarts from the new tokens
        }
    }
    const slot_file_part dft = w.write(buf.data(), buf.size());
    std::vector<uint8_t> spec_state;
    common_speculative_get_state(s.spec, s.id, spec_state);
    const slot_file_part spec = w.write(spec_state.data(), spec_state.size());
    buf = {};

    json ckj = json::array();
    for (const auto * c : file_ckpts(*s.prompt, n_tok, st, base)) {
        const slot_file_part ct = w.write(c->data_tgt.data(), c->data_tgt.size());
        const slot_file_part cd = w.write(c->data_dft.data(), c->data_dft.size());
        const slot_file_part cs = w.write(c->data_spec.data(), c->data_spec.size());
        ckj.push_back({ { "n_tokens", c->n_tokens }, { "pos_min", c->pos_min }, { "pos_max", c->pos_max },
                        { "tgt", part_json(ct) }, { "dft", part_json(cd) }, { "spec", part_json(cs) } });
    }
    const slot_file_part tp = w.write(packed.data(), packed.size());

    const json mj = {
        { "model",       model_fp },
        { "base",        base },
        // the base is not a turn: a new conversation starting from it begins its own
        { "prompt_n",    base ? 0 : st.prompt_n },
        { "turn_n",      base ? 0 : st.turn_n },
        { "tgt",         part_json(tgt) },
        { "dft",         part_json(dft) },
        { "spec",        part_json(spec) },
        { "tokens",      part_json(tp) },
        { "checkpoints", ckj },
    };
    const std::string m = mj.dump();
    const uint64_t meta_off = w.off, meta_size = m.size(), meta_hash = hash_bytes(m.data(), m.size());
    w.f.write(m.data(), m.size());
    w.f.write((const char *) &meta_off, 8).write((const char *) &meta_size, 8).write((const char *) &meta_hash, 8);
    w.f.write((const char *) &SLOT_FILE_VERSION, 4).write((const char *) &SLOT_FILE_MAGIC, 4);
    w.f.close();
    if (!w.f) {
        return fail("write failed (disk full?)");
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        return fail("rename of " + tmp + " failed: " + ec.message());
    }
    res.ok       = true;
    res.n_tokens = n_tok;
    res.n_bytes  = meta_off + meta_size + SLOT_FILE_FOOTER;
    res.ms       = (ggml_time_us() - t0) / 1000.0;
    return res;
}

// Loads the file into the slot. If it fails, the slot stays empty: never a half state.
static slot_file_result slot_file_load(slot_file_view & s, const std::string & path, uint64_t model_fp, slot_disk_state & st) {
    const int64_t t0 = ggml_time_us();
    slot_file_result res;

    // the slot is emptied first: if loading stops halfway, nothing old or new is left
    auto wipe = [&]() {
        llama_memory_seq_rm(llama_get_memory(s.ctx_tgt), s.id, -1, -1);
        if (s.ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(s.ctx_dft), s.id, -1, -1);
        }
        common_speculative_set_state(s.spec, s.id, {});
        s.prompt->clear();
    };
    wipe();

    slot_file_meta meta;
    if (!slot_file_read_meta(path, meta, res.error)) {
        return res;
    }
    if (meta.model != model_fp) {
        res.error = "the file comes from another model or another cache configuration";
        return res;
    }
    server_tokens toks;
    if (!read_tokens(path, meta.tokens, s.has_mtmd, toks, res.error)) {
        return res;
    }
    if (!toks.validate(s.ctx_tgt)) {
        res.error = "tokens not valid for this model";
        return res;
    }

    std::ifstream f(path, std::ios::binary);
    std::list<common_prompt_checkpoint> ckpts;
    for (const auto & cm : meta.ckpts) {
        common_prompt_checkpoint c;
        c.update_pos(cm.n_tokens, cm.pos_min, cm.pos_max);
        if (!read_part(f, cm.tgt, c.data_tgt) || !read_part(f, cm.dft, c.data_dft) || !read_part(f, cm.spec, c.data_spec)) {
            res.error = "corrupted checkpoint (pos " + std::to_string(cm.pos_max) + ")";
            return res;
        }
        ckpts.push_back(std::move(c));
    }

    std::vector<uint8_t> buf;
    if (!read_part(f, meta.tgt, buf)) {
        res.error = "corrupted state: the hash does not match";
        return res;
    }
    if (llama_state_seq_set_data_ext(s.ctx_tgt, buf.data(), buf.size(), s.id, LLAMA_STATE_SEQ_FLAGS_NONE) != buf.size()) {
        wipe();
        res.error = "the state does not load (context too small or file from another configuration)";
        return res;
    }
    buf = {};

    // The draft only improves speed: if it is missing or does not load, it restarts from the new tokens and the answer does not change.
    if (s.ctx_dft) {
        std::vector<uint8_t> dft, spec;
        const bool ok = meta.dft.size > 0 && read_part(f, meta.dft, dft) && read_part(f, meta.spec, spec) &&
                        llama_state_seq_set_data_ext(s.ctx_dft, dft.data(), dft.size(), s.id, LLAMA_STATE_SEQ_FLAGS_NONE) == dft.size();
        if (ok) {
            common_speculative_set_state(s.spec, s.id, spec);
        } else {
            SRV_WRN("%s: draft state not loaded: drafts restart from the new tokens\n", path.c_str());
            llama_memory_seq_rm(llama_get_memory(s.ctx_dft), s.id, -1, -1);
        }
    }

    s.prompt->tokens      = std::move(toks);
    s.prompt->checkpoints = std::move(ckpts);
    st.prompt_n = meta.prompt_n;
    st.turn_n   = meta.turn_n;

    res.ok       = true;
    res.n_tokens = s.prompt->tokens.size();
    res.n_bytes  = fs::file_size(path);
    res.ms       = (ggml_time_us() - t0) / 1000.0;
    return res;
}

// ---- disk cache ----

// How many tokens of `prompt` are reused starting from a state with these tokens and checkpoints. With the
// recurrent state: the whole state if it is a prefix of the prompt (at least one token must be read to get the
// logits), otherwise the most advanced checkpoint inside the common prefix, nothing in between. With attention
// only, the state is shortened down to the common prefix.
static size_t reusable(const server_tokens & state, const std::vector<int64_t> & ckpt_n_tokens, const server_tokens & prompt,
                       bool can_truncate) {
    if (state.empty() || prompt.empty()) {
        return 0;
    }
    const size_t limit = std::min(state.get_common_prefix(prompt), prompt.size() - 1);
    if (state.size() <= limit || can_truncate) {
        return std::min(state.size(), limit);
    }
    size_t best = 0;
    for (const int64_t n : ckpt_n_tokens) {
        if ((size_t) n <= limit) {
            best = std::max(best, (size_t) n);
        }
    }
    return best;
}

static std::vector<int64_t> ckpt_n_tokens(const server_prompt & p) {
    std::vector<int64_t> out;
    for (const auto & c : p.checkpoints) {
        out.push_back(c.n_tokens);
    }
    return out;
}

slot_disk_cache::slot_disk_cache(std::string dir, size_t limit_bytes, int64_t max_age_s, uint64_t model_fp, bool can_truncate)
    : dir(std::move(dir)), limit_bytes(limit_bytes), max_age_s(max_age_s), model_fp(model_fp), can_truncate(can_truncate) {
    fs::create_directories(this->dir);
}

// Index entry from the file's metadata and tokens (without reading the state).
static bool read_entry(const std::string & path, uint64_t model_fp, slot_disk_entry & e, std::string & err) {
    slot_file_meta meta;
    if (!slot_file_read_meta(path, meta, err)) {
        return false;
    }
    if (meta.model != model_fp) {
        err = "comes from another model or another cache configuration";
        return false;
    }
    // the index compares only tokens: images count by their identifier, which the file carries
    if (!read_tokens(path, meta.tokens, true, e.tokens, err)) {
        return false;
    }
    e.path = path;
    e.ckpt_n_tokens.clear();
    for (const auto & c : meta.ckpts) {
        e.ckpt_n_tokens.push_back(c.n_tokens);
    }
    e.base   = meta.base;
    e.bytes  = fs::file_size(path);
    e.t_used = file_time_s(fs::last_write_time(path));
    return true;
}

void slot_disk_cache::scan() {
    entries.clear();
    const int64_t now = now_s();
    for (const auto & de : fs::directory_iterator(dir)) {
        if (!de.is_regular_file()) {
            continue;
        }
        const std::string path = de.path().string();
        const std::string ext  = de.path().extension().string();
        std::error_code ec;
        if (ext == ".tmp") {
            SRV_WRN("disk cache: removing %s, half written (shutdown during a save)\n", path.c_str());
            fs::remove(path, ec);
            continue;
        }
        if (ext != ".slot") {
            continue;
        }
        slot_disk_entry e;
        std::string err;
        if (!read_entry(path, model_fp, e, err)) {
            SRV_WRN("disk cache: removing %s: %s\n", path.c_str(), err.c_str());
            fs::remove(path, ec);
            continue;
        }
        if (now - e.t_used > max_age_s) {
            SRV_INF("disk cache: removing %s, unused for %lld hours\n", path.c_str(), (long long) ((now - e.t_used) / 3600));
            fs::remove(path, ec);
            continue;
        }
        entries.push_back(std::move(e));
    }
    SRV_INF("disk cache: %zu states on disk, %.2f GiB (cap %.2f GiB, %lld hours) in %s\n", entries.size(),
            total_bytes() / 1073741824.0, limit_bytes / 1073741824.0, (long long) (max_age_s / 3600), dir.c_str());
}

size_t slot_disk_cache::total_bytes() const {
    size_t s = 0;
    for (const auto & e : entries) {
        s += e.bytes;
    }
    return s;
}

void slot_disk_cache::drop(size_t i, const char * why) {
    SRV_INF("disk cache: removing %s (%zu tokens, %.2f GiB): %s\n", entries[i].path.c_str(), entries[i].tokens.size(),
            entries[i].bytes / 1073741824.0, why);
    std::error_code ec;
    fs::remove(entries[i].path, ec);
    entries.erase(entries.begin() + i);
}

void slot_disk_cache::expire() {
    const int64_t now = now_s();
    for (size_t i = entries.size(); i-- > 0;) {
        if (now - entries[i].t_used > max_age_s) {
            drop(i, "unused for longer than the age limit");
        }
    }
}

// Makes room for a state of `need` bytes, within the cap and leaving DISK_MIN_FREE free on the disk. It first checks
// that removing everything removable would make it fit (otherwise it removes nothing), then removes the least
// recently used conversations and only after that the bases, which weigh little and serve every new conversation.
bool slot_disk_cache::make_room(size_t need, const std::string & keep) {
    expire();
    std::error_code ec;
    const auto sp = fs::space(dir, ec);
    if (ec) {
        SRV_ERR("disk cache: free space on %s unreadable: %s\n", dir.c_str(), ec.message().c_str());
        return false;
    }
    size_t kept = 0, freeable = 0;
    for (const auto & e : entries) {
        (e.path == keep ? kept : freeable) += e.bytes;
    }
    if (kept + need > limit_bytes || sp.available + freeable < need + DISK_MIN_FREE) {
        SRV_WRN("disk cache: a %.2f GiB state does not fit (cap %.2f GiB, %.2f GiB free on the disk, %.0f always stay free): not saved\n",
                need / 1073741824.0, limit_bytes / 1073741824.0, sp.available / 1073741824.0, DISK_MIN_FREE / 1073741824.0);
        return false;
    }
    size_t available = sp.available;
    while (total_bytes() + need > limit_bytes || available < need + DISK_MIN_FREE) {
        size_t victim = entries.size();
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto & e = entries[i];
            if (e.path == keep) {
                continue;
            }
            if (victim == entries.size()) {
                victim = i;
                continue;
            }
            const auto & v = entries[victim];
            if (e.base != v.base ? !e.base : e.t_used < v.t_used) {
                victim = i;
            }
        }
        available += entries[victim].bytes;
        drop(victim, total_bytes() + need > limit_bytes ? "space cap" : "disk almost full");
    }
    return true;
}

slot_file_result slot_disk_cache::write_entry(const slot_file_view & s, const std::string & path, bool base, size_t n_tok,
                                              const slot_disk_state & st, const std::string & keep) {
    slot_file_result res;
    if (!make_room(state_size(s, n_tok, st, base), keep)) {
        res.error = "not enough space";
        return res;
    }
    res = save_n(s, path, model_fp, base, st, n_tok);
    std::string err;
    slot_disk_entry e;
    if (res.ok && !read_entry(path, model_fp, e, err)) {
        res.ok = false;
        res.error = "the file just written cannot be read back: " + err;
    }
    if (!res.ok) {
        SRV_ERR("disk cache: saving %zu tokens failed: %s\n", n_tok, res.error.c_str());
        std::error_code ec;
        fs::remove(path, ec);
        return res;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].path == path) {
            entries.erase(entries.begin() + i);
            break;
        }
    }
    entries.push_back(std::move(e));
    SRV_INF("disk cache: saved %s of %zu tokens in %s, %.2f GiB in %.0f ms\n", base ? "the base (system prompt and tools)" : "the conversation",
            res.n_tokens, path.c_str(), res.n_bytes / 1073741824.0, res.ms);
    return res;
}

slot_file_result slot_disk_cache::persist(const slot_file_view & s, slot_disk_state & st, const std::string & keep) {
    slot_file_result res;
    const server_tokens & cur = s.prompt->tokens;
    if (cur.empty()) {
        return res;
    }
    const uint64_t h = tokens_hash(cur);
    if (h == st.hash || h == failed_hash) {
        return res; // already on disk as it is, or there was no room for this same state
    }
    // how much it would cost to rebuild it from the files that are already there
    size_t covered = 0;
    for (const auto & e : entries) {
        covered = std::max(covered, reusable(e.tokens, e.ckpt_n_tokens, cur, can_truncate));
    }
    if (cur.size() < covered + DISK_MIN_SAVE) {
        return res;
    }
    // the previous saves of the same conversation are contained in this one: remove them (the base stays)
    for (size_t i = entries.size(); i-- > 0;) {
        const auto & e = entries[i];
        if (!e.base && e.path != keep && e.tokens.get_common_prefix(cur) == e.tokens.size()) {
            drop(i, "superseded by a more recent save of the same conversation");
        }
    }
    char name[32];
    snprintf(name, sizeof(name), "%016llx.slot", (unsigned long long) h);
    res = write_entry(s, (fs::path(dir) / name).string(), false, cur.size(), st, keep);
    if (!res.ok) {
        failed_hash = h;
        return res;
    }
    st.hash = h;
    return res;
}

bool slot_disk_cache::restore(slot_file_view & s, slot_disk_state & st, const std::string & path) {
    size_t i = 0;
    while (i < entries.size() && entries[i].path != path) {
        ++i;
    }
    if (i == entries.size()) {
        return false;
    }
    const slot_file_result res = slot_file_load(s, path, model_fp, st);
    if (!res.ok) {
        SRV_ERR("disk cache: restoring %s failed, rereading from scratch: %s\n", path.c_str(), res.error.c_str());
        drop(i, "file rejected");
        st.hash = 0;
        st.prompt_n = 0;
        st.turn_n = 0;
        return false;
    }
    std::error_code ec;
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);
    entries[i].t_used = now_s();
    st.hash = tokens_hash(s.prompt->tokens);
    SRV_INF("disk cache: restored %zu tokens from %s (%.2f GiB) in %.0f ms\n", res.n_tokens, path.c_str(),
            res.n_bytes / 1073741824.0, res.ms);
    return true;
}

bool slot_disk_cache::base_worth(size_t base_n, size_t prompt_n) {
    return base_n >= DISK_MIN_BASE && base_n + 1 < prompt_n;
}

void slot_disk_cache::prepare(slot_file_view & s, slot_disk_state & st, const server_tokens & prompt, size_t base_n) {
    st.base_pos = -1;
    if (prompt.empty()) {
        return;
    }
    const server_prompt & cur = *s.prompt;
    const size_t own = reusable(cur.tokens, ckpt_n_tokens(cur), prompt, can_truncate);
    std::string best;
    size_t best_n = 0, best_bytes = 0;
    for (const auto & e : entries) {
        const size_t n = reusable(e.tokens, e.ckpt_n_tokens, prompt, can_truncate);
        if (n > best_n || (n == best_n && n > 0 && e.bytes < best_bytes)) {
            best = e.path;
            best_n = n;
            best_bytes = e.bytes;
        }
    }
    const bool swap = !best.empty() && best_n >= own + DISK_MIN_GAIN;
    // The state in the slot is about to be replaced, or overwritten before the end of the last prompt (another
    // conversation): it goes to disk first. The answer after the prompt matters little: the next turn of the same
    // conversation rereads it anyway if it renders it without the reasoning.
    const size_t valuable = st.prompt_n > 0 ? std::min(cur.tokens.size(), (size_t) st.prompt_n) : cur.tokens.size();
    if (swap || valuable >= own + DISK_MIN_SAVE) {
        persist(s, st, best);
    }
    if (swap) {
        SRV_INF("disk cache: the file reuses %zu tokens of the prompt, the slot %zu\n", best_n, own);
        restore(s, st, best);
    }

    // the common prefix is not on disk yet: it is saved when the prompt read reaches it
    uint64_t h = 0;
    if (!base_worth(base_n, prompt.size()) || !base_hash(prompt, base_n, h)) {
        return;
    }
    std::error_code ec;
    if (fs::is_regular_file(fs::path(dir) / base_file_name(h), ec)) {
        return;
    }
    st.base_pos = (int32_t) base_n;
}

void slot_disk_cache::save_base(const slot_file_view & s, slot_disk_state & st) {
    const int32_t pos = st.base_pos;
    st.base_pos = -1;
    uint64_t h = 0;
    const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(s.ctx_tgt), s.id);
    if ((int32_t) s.prompt->tokens.size() < pos || pos_max + 1 != s.prompt->tokens.pos_next(pos) || !base_hash(s.prompt->tokens, pos, h)) {
        SRV_WRN("disk cache: base not saved, the memory reaches %d instead of %d\n", pos_max + 1, pos);
        return;
    }
    write_entry(s, (fs::path(dir) / base_file_name(h)).string(), true, (size_t) pos, st, "");
}

bool slot_disk_cache::base_on_disk(const server_tokens & prompt, size_t base_n) const {
    uint64_t h = 0;
    if (!base_worth(base_n, prompt.size()) || !base_hash(prompt, base_n, h)) {
        return false;
    }
    std::error_code ec;
    return fs::is_regular_file(fs::path(dir) / base_file_name(h), ec);
}

void slot_disk_cache::restore_latest(slot_file_view & s, slot_disk_state & st) {
    const slot_disk_entry * latest = nullptr;
    for (const auto & e : entries) {
        if (!latest || e.t_used > latest->t_used) {
            latest = &e;
        }
    }
    if (latest) {
        const std::string path = latest->path; // restore() may remove the entry from the index
        restore(s, st, path);
    }
}
