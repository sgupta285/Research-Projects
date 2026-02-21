from __future__ import annotations
import argparse, os
import requests

HOTPOT_URLS = {
  "train":"http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
  "dev_distractor":"http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
  "dev_fullwiki":"http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
}

def _download(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path,"wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk: f.write(chunk)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa","legalbenchrag"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev_distractor")
    ap.add_argument("--url", default=None)
    args=ap.parse_args()

    if args.dataset=="hotpotqa":
        url=HOTPOT_URLS[args.split]
        out_file=os.path.join(args.out, os.path.basename(url))
        print(f"[DL] {url} -> {out_file}")
        _download(url, out_file)
        print("[OK] Done.")
        return

    if not args.url:
        raise SystemExit("For legalbenchrag, pass --url with the download link (see official repo README).")
    out_file=os.path.join(args.out, "legalbenchrag_download.zip")
    print(f"[DL] {args.url} -> {out_file}")
    _download(args.url, out_file)
    print("[OK] Downloaded zip. Unzip it so you have corpus/ and benchmarks/ under data/legalbenchrag.")
if __name__=="__main__":
    main()
