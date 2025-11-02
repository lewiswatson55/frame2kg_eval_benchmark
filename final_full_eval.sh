
frame2kg-eval --pred-dir /Users/watson/PycharmProjects/frame2kg_eval_codebase/preds_full/3B-QKVO/4 --gt hf:lewiswatson/Frame2KG-YC2:testing --out ./results_final/3B-QKVO_final.csv
frame2kg-eval --pred-dir /Users/watson/PycharmProjects/frame2kg_eval_codebase/preds_full/3B-QKVO-Gate/4 --gt hf:lewiswatson/Frame2KG-YC2:testing --out ./results_final/3B-QKVO-Gate_final.csv
frame2kg-eval --pred-dir /Users/watson/PycharmProjects/frame2kg_eval_codebase/preds_full/7B-QKVO/3 --gt hf:lewiswatson/Frame2KG-YC2:testing --out ./results_final/7B-QKVO_best.csv
frame2kg-eval --pred-dir /Users/watson/PycharmProjects/frame2kg_eval_codebase/preds_full/7B-QKVO-Gate/1 --gt hf:lewiswatson/Frame2KG-YC2:testing --out ./results_final/7B-QKVO-Gate_step1k.csv

