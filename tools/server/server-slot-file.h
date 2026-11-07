#pragma once

// Disk cache of conversations (--cache-disk): a slot's state stays on disk across restarts and when
// switching between conversations, so the engine only rereads the new tokens.
//
// A file contains, in this order:
//   1. the model state for the sequence (KV of the attention layers and the recurrent state at the end);
//   2. the state of the draft context (MTP or draft model), if there is one: optional, without it the
//      drafts restart from the new tokens and the answer does not change (every draft is verified);
//   3. the useful checkpoints: the recurrent state at earlier points. The recurrent state cannot be
//      rewound, so a prompt that diverges before the end (the next turn renders the answer without the
//      reasoning) restarts from the end-of-prompt checkpoint instead of from zero;
//   4. the tokens (with the images, if there are any);
//   5. the metadata (JSON) with the hash of the model and of every part;
//   6. a 32-byte footer that locates the metadata.
// It is written to <file>.tmp and renamed at the end: a half-written file never has the real name. On
// restore every hash is checked again: a broken file, or one from another model, is rejected.

#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <cstdint>
#include <string>
#include <vector>

struct slot_file_result {
    bool        ok = false;
    std::string error;  // why it failed, for the log and for the HTTP response
    size_t      n_tokens = 0;
    size_t      n_bytes  = 0;
    double      ms       = 0.0;
};

// What the disk cache touches of a slot. `prompt` holds the cached tokens and the slot's checkpoints.
struct slot_file_view {
    llama_context *      ctx_tgt = nullptr;
    llama_context *      ctx_dft = nullptr;
    common_speculative * spec    = nullptr;
    llama_seq_id         id      = 0;
    bool                 has_mtmd = false;
    server_prompt *      prompt  = nullptr;
};

// Disk cache state per slot.
struct slot_disk_state {
    uint64_t hash     = 0;   // hash of the tokens at the last save or restore
    int32_t  base_pos = -1;  // end of the common prefix to save as the base when the prompt read reaches it, -1 = none
    int32_t  prompt_n = 0;   // prompt length of the last request
    // Prompt length of the first request of the turn (a user message, then the steps with tools that
    // extend it). The next turn renders the answers of the whole turn without reasoning and diverges there: the
    // checkpoints up to that point are the ones that are needed, in the file and in the slot's memory.
    int32_t  turn_n   = 0;
};

// The state checkpoints not to lose: the two most advanced ones up to the start of the turn (see turn_n).
bool slot_disk_protected(const server_prompt & p, const slot_disk_state & st, const common_prompt_checkpoint & c);

// How many tokens of the prompt are the common prefix (`base_text`: the text before the first user message,
// see chat_base_prefix), 0 if the prompt has none.
size_t slot_disk_base_length(const llama_vocab * vocab, const std::string & base_text, const server_tokens & prompt);

// Hash of what makes a state reusable: model and image encoder files (size, first and last 4 MiB),
// LoRA adapters, cache types, format version.
uint64_t slot_file_model_fingerprint(const common_params & params);

// A disk cache file, identified by its tokens (not by who is talking).
struct slot_disk_entry {
    std::string          path;
    server_tokens        tokens;
    std::vector<int64_t> ckpt_n_tokens;  // where the file's checkpoints are
    bool                 base   = false;
    size_t               bytes  = 0;
    int64_t              t_used = 0;     // seconds, last save or restore
};

// Before every request the prompt is compared with the state in the slot and with the files: if a file reuses
// more of it, the current state is saved (if it is not already) and the file is loaded. The common prefix
// (system prompt + tools) is saved as a "base" the first time it is read: a new conversation starts from there.
struct slot_disk_cache {
    std::string dir;
    size_t      limit_bytes;
    int64_t     max_age_s;
    uint64_t    model_fp;
    bool        can_truncate;  // the model memory can be shortened freely (attention only, no recurrent state)
    std::vector<slot_disk_entry> entries;
    uint64_t    failed_hash = 0;  // a state that did not fit: not retried until it changes

    slot_disk_cache(std::string dir, size_t limit_bytes, int64_t max_age_s, uint64_t model_fp, bool can_truncate);

    // Index the folder: drop leftover .tmp files and files that are expired, broken or from another model.
    void scan();

    // Before a completion request on the slot (see above). `base_n`: where the common prefix ends in the
    // prompt (start of the first user message), 0 if unknown.
    void prepare(slot_file_view & s, slot_disk_state & st, const server_tokens & prompt, size_t base_n);

    // Save the slot if it changed and rereading it would be costly: `ok` false with empty `error` = nothing to do.
    // `keep` is not deleted even if the new state contains it.
    slot_file_result persist(const slot_file_view & s, slot_disk_state & st, const std::string & keep = "");

    // The slot is at the end of the common prefix (st.base_pos): save it as the base.
    void save_base(const slot_file_view & s, slot_disk_state & st);

    // At startup: load the most recently used state (the conversation open at shutdown).
    void restore_latest(slot_file_view & s, slot_disk_state & st);

    // Is the base of this prompt already on disk? Called from HTTP threads: it only looks at the file name.
    bool base_on_disk(const server_tokens & prompt, size_t base_n) const;

    // Is the common prefix worth a file? (long enough and with something left to read after it)
    static bool base_worth(size_t base_n, size_t prompt_n);

    static constexpr int64_t IDLE_SAVE_S = 30;  // slot idle for long enough: save it

private:
    bool restore(slot_file_view & s, slot_disk_state & st, const std::string & path);
    slot_file_result write_entry(const slot_file_view & s, const std::string & path, bool base, size_t n_tok,
                                 const slot_disk_state & st, const std::string & keep);
    bool make_room(size_t need, const std::string & keep);
    void drop(size_t i, const char * why);
    void expire();
    size_t total_bytes() const;
};
