import re
import os

def parse_logs():
    base_path = "./"
    log_files = [
        "block_floating_point/run_eval_bfp_tinyllama_128.log",
        "block_floating_point/run_eval_bfp_tinyllama_64.log",
        "block_floating_point/run_eval_bfp_tinyllama_32.log",
        "fix_precision_awq/run_eval_awq_tinyllama_128.log",
        "fix_precision_awq/run_eval_awq_tinyllama_64.log",
        "mix_precision_awq/run_eval_awq_tinyllama_64.log",
    ]

    results = []

    for rel_path in log_files:
        full_path = os.path.join(base_path, rel_path)
        if not os.path.exists(full_path):
            continue

        if "bfp" in rel_path:
            method = "BFP"
        elif "mix" in rel_path:
            method = "AWQ(Mix)"
        else:
            method = "AWQ(Fix)"

        current_config = None

        with open(full_path, "r") as f:
            for line in f:
                # Detect Method & Bits (AWQ or BFP)
                # Group 1: Bits, Group 2: H, Group 3: W
                m = re.search(r"Mantissa Bits = (\d+), Block (?:Height = (\d+), Block Width = (\d+)|Size = (\d+)x(\d+))", line)
                if not m:
                    # Alternative format
                    m = re.search(r"Running.*Mantissa Bits = (\d+),.*Block Size = (\d+)x(\d+)", line)
                
                if m:
                    if current_config:
                        results.append(current_config)
                    
                    bits = m.group(1)
                    if m.group(2) and m.group(3):
                        h, w = m.group(2), m.group(3)
                    elif m.group(4) and m.group(5):
                        h, w = m.group(4), m.group(5)
                    else:
                        h, w = m.group(2), m.group(3) # Fallback to first pair if regex groups shifted
                    
                    current_config = {
                        "method": method,
                        "bits": bits,
                        "h": h,
                        "w": w,
                        "ppl": "N/A",
                        "acc_norm": "N/A"
                    }
                    continue

                # Detect PPL
                m = re.search(r"(?:WikiText2|wikitext) PPL: ([\d\.]+)", line)
                if m and current_config:
                    current_config["ppl"] = m.group(1)

                # Detect Accuracy (support both acc and acc_norm)
                m = re.search(r"acc_norm\): ([\d\.]+)", line)
                if m and current_config:
                    current_config["acc_norm"] = m.group(1)
            
            # Add last one
            if current_config:
                results.append(current_config)

    # Unique check based on (method, bits, h, w)
    unique_results = {}
    for r in results:
        key = (r["method"], r["bits"], r["h"], r["w"])
        # Update if we have better data (non-N/A)
        if key not in unique_results:
            unique_results[key] = r
        else:
            if r["ppl"] != "N/A": unique_results[key]["ppl"] = r["ppl"]
            if r["acc_norm"] != "N/A": unique_results[key]["acc_norm"] = r["acc_norm"]

    # Sort results
    def sort_key(x):
        try:
            total_size = int(x["h"]) * int(x["w"])
        except:
            total_size = 9999
        return (total_size, x["method"], -int(x["bits"]), int(x["h"]))

    sorted_results = sorted(unique_results.values(), key=sort_key)

    print("| Method | Block Size | Bits | Avg Bits/W¹ | WikiText2 PPL | HellaSwag (acc_norm) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in sorted_results:
        size = f"{r['h']}x{r['w']}"
        try:
            block_size = int(r["h"]) * int(r["w"])
            avg_bits = int(r["bits"]) + (8.0 / block_size)
            avg_bits_str = f"{avg_bits:.3f}"
        except:
            avg_bits_str = "N/A"
            
        print(f"| {r['method']} | {size} | {r['bits']} | {avg_bits_str} | {r['ppl']} | {r['acc_norm']} |")

if __name__ == "__main__":
    parse_logs()
