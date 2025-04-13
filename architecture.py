ROBOT_TASK_RECOGNITION_RL/                  ← Main project folder
│
├── notebooks/                              ← Global results and visualizations
│   ├── agent1_achievements.ipynb
│   ├── agent2_achievements.ipynb
│   └── agent_1&2_achievements.ipynb
│
├── agent1_patterns_chests_to_reach/        ← Agent 1: Event-to-chest task
│   ├── approach1_simple_event_mapping/     ← Approach 1: One-to-one mapping
│   │   ├── agents/
│   │   │   ├── base_agent.py
│   │   │   ├── ppo_agent.py
│   │   │   └── dqn_agent.py
│   │   ├── training/
│   │   │   └── train_agent.py
│   │   ├── weights/
│   │   ├── logs/
│   │   └── evaluate_agent.py
│   │
│   ├── approach2_temporal_window/          ← Approach 2: Fixed-length memory
│   │   ├── agents/
│   │   ├── training/
│   │   ├── weights/
│   │   └── logs/
│   │
│   ├── approach3_advanced_sequence_modeling/  ← Approach 3: LSTM / Transformer
│   │   ├── agents/
│   │   │   ├── lstm_agent.py
│   │   │   └── transformer_agent.py
│   │   ├── training/
│   │   ├── weights/
│   │   └── logs/
│   │
│   ├── env/
│   │   ├── register_envs.py
│   │   └── wrappers.py
│   │
│   ├── utils/
│   │   ├── event_encoding.py
│   │   └── visualization.py
│   │
│   └── config/
│       └── agent1_ppo_simple.yaml
│
├── agent2_physically_reach_chest/          ← Agent 2: PyBullet robot
│   ├── env/
│   │   └── kuka_arm_env.py
│   ├── training/
│   │   └── train_arm_agent.py
│   ├── weights/
│   ├── logs/
│   └── utils/
│       └── motion_encoding.py
│
├── debug/                                  ← Scripts for manual tests & debug
│   └── debug_env_interaction.py
│
├── requirements.txt                        ← Python dependencies
└── README.md                               ← Project description
