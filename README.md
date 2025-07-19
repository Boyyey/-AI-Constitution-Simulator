# 🧠💡 AI Constitution Simulator

## 🚀 What is it?
A PhD-level, all-in-one Streamlit app to generate, simulate, and compare constitutions and governance frameworks. Test-drive new social contracts in a virtual society with advanced agent-based modeling, deep analytics, and beautiful visualizations.

---

## ✨ Features
- 📝 **Constitution Generation**: Use OpenAI (optional) or a mock generator to create constitutions from prompts.
- 🧩 **Agent-Based Simulation**: Citizens, politicians, judges, journalists, and media with evolving beliefs, social networks, and dynamic events.
- 📊 **Deep Analytics**: Track inequality, unrest, happiness, polarization, corruption, press freedom, protests, and more.
- 🕸️ **Social Network Visualization**: See how ideology and influence spread.
- 📉 **Constitution Comparison**: Run and rank multiple constitutions, export results.
- 🖥️ **Modern UI**: Beautiful, interactive Streamlit dashboard.
- 🔑 **OpenAI API Integration**: Optional—add your API key for real LLM-powered constitutions.

---

## 🏁 Quickstart
1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run ui.py
   ```
3. **(Optional) Add your OpenAI API key** in the sidebar for real LLM generation.

---

## 🛠️ Usage Tips
- **Generate** a constitution with your own prompt or use the default.
- **Simulate** a society with adjustable parameters (population, years, agent types).
- **Analyze** outcome metrics and visualize trends, unrest, and social networks.
- **Compare** different constitutions and export results as CSV.
- **All logic is in `ui.py`.** `main.py` is deprecated and will only show a message.

---

## 🧬 Tech Stack
- **Python** (3.8+ recommended)
- **Streamlit** for UI
- **NumPy, Pandas, Matplotlib, Seaborn** for analytics
- **NetworkX** for social network graphs
- **OpenAI** (optional, for LLM generation)

---

## 🐞 Troubleshooting
- If you see errors about missing packages, run:
  ```bash
  pip install -r requirements.txt
  ```
- If you want to use OpenAI, install the package:
  ```bash
  pip install openai
  ```
- For best performance, use Chrome or Firefox.

---

## 🤝 Contributing
- Pull requests and issues welcome!
- Ideas for new agent types, metrics, or visualizations? Open an issue or PR.

---

## 📜 License
MIT

---

## 🌍 Who is this for?
- Political scientists, NGOs, educators, policy thinkers, and anyone curious about governance.
- Use it for research, teaching, or just to explore wild new social contracts!

---

## 💡 Example Prompts
- "Draft a constitution maximizing freedom and stability."
- "What if we had direct democracy, UBI, and no supreme court?"
- "Design a governance framework for a post-conflict society."

---

## 🏆 Why it’s next-level
- Combines agent-based simulation, NLP, political science, and data science.
- Lets you discover non-obvious trade-offs and novel governance designs.
- All in a single, beautiful, interactive app.

---

## 🏁 Get started now:
```bash
streamlit run ui.py
``` 