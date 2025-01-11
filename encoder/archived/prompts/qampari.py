# prompts for answering
instruction_prompt = "Answer the given question using the provided search results (some of which might be irrelevant)."
doc_prompt_template = "[{ID}]: (Title: {T}) {P}\n"
inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch results:\n{D}\n{PREFIX}"
demo_sep = "\n\n"

# promtps for asking feedback
## with answer 
# TBD

## without answer 
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch result: {D}\n{PREFIX}"
fbk_instruction_prompt = "Read and understand the given question. Then, based on the question, identify the useful information in the provided search results (some of which might be irrelevant)."

### (1) Instruction first + Query rewriting
# fbk_prefix="Rewritten question: "
# fbk_instruction_prompt += " Finally, rewrite the question for searching additional new documents. These new documents are expected to complete the missing knowledge about the question." 

### (2) Instruction first + Query expansion
# fbk_prefix="New keyword combintation:"
# fbk_instruction_prompt += " Finally, write a new keyword combintation for searching additional new documents. These new documents are expected to complete the missing knowledge about the question." 

### prompt separately
### (3) Instruction later + Query rewriting
# fbk_prefix="Based on the identified useful information, rewrite the question for searching additional additional new documents. These new documents are expected to complete the missing knowledge about the question.\n\nRewritten question:"

### (4) Instruction later + Query expansion
fbk_prefix="Based the identified useful information, write a new new keyword combination for searching additional new documents. These new documents are expected to complete the missing knowledge about the question.\n\nNew keyword combination: "

def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item.get('title', ''))
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

# def apply_demo_prompt(Q, D, A, instruction=""):
#     p = demo_prompt_template
#     p = p.replace("{INST}", instruction).strip()
#     p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
#     return p

def apply_inst_prompt(Q, D, instruction="", prefix=""):
    p = inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", "")
    p = p.replace("{PREFIX}", prefix).strip()
    return p

def apply_fbk_inst_prompt(Q, D, instruction="", prefix=""):
    p = fbk_inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
