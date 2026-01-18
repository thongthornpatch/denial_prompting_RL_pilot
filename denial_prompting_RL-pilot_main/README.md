# Denial Prompting in RL training

## Project Structure
```
denial_prompting_RL/
├── configs/             
│   ├── config_laptop.yaml  # Laptop testing config
│   └── config_nscc.yaml    # NSCC training config (GPU)
├── src/
│   ├── data/              
│   ├── models/             # Model wrappers
│   ├── training/           # GRPO training loop
│   ├── rewards/            # Implementation of reward function (Reward = Correctness - (num_violations × penalty_weight))
│   ├── evaluation/         # Pass@k and NeoGauge metrics
│   └── utils/             
├── data/
│   ├── raw/                # Raw NeoCoder dataset
│   └── processed/          # Processed data
├── scripts/                
├── notebooks/              # Jupyter notebooks for Google colab
├── experiments/            # Saved experiment results
├── logs/                   # Training logs
└── requirements.txt        
```


