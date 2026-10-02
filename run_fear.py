import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

bundle = torch.load("fear_dir.pt", map_location="cpu")
MODEL_ID = bundle["model_id"]
layer_index = bundle["layer_index"]
fear_dir = bundle["fear_dir"]
DOSES = [0, 2, 4, 6, 8]
MAX_NEW = 80
PROMPT = "Describe what you feel right now."

def hook(_module, _inp, out, state=None):
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone()
    h[:, -1, :] = h[:, -1, :] + state["dose"] * fear_dir.to(h.dtype)
    return (h,) + out[1:] if isinstance(out, tuple) else h

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    layer = model.model.layers[layer_index]
    state = {"dose": 0.0}
    handle = layer.register_forward_hook(lambda m, i, o: hook(m, i, o, state))
    ids = tok(PROMPT, return_tensors="pt")
    try:
        for dose in DOSES:
            state["dose"] = float(dose)
            with torch.no_grad():
                gen = model.generate(
                    **ids, max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            print({"dose": dose, "output": text}, flush=True)
    finally:
        handle.remove()
    print("process-only: hook dies with this process; no experience claimed", flush=True)

if __name__ == "__main__":
    main()
