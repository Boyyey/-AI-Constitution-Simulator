# 🧠💡 AI Constitution Simulator (Modular, PhD-level Streamlit App)
# ---------------------------------------------------------------
# This file is the main dashboard UI. All logic is now modularized.
# Just run: streamlit run ui.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional

from society import SocietyModel
from constitution_parser import parse_constitution_text
from analytics import plot_tradeoff, export_csv, export_json, export_pdf, plot_agent_timeline
from optimizer import optimize_constitution

# --- Section 1: Constitution Generation ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def generate_constitution(prompt, api_key=None):
    """Generate a constitution using OpenAI API if available, else return a mock."""
    if api_key and OPENAI_AVAILABLE:
        try:
            openai.api_key = api_key
            chat_completion = getattr(openai, "ChatCompletion", None)
            completion = getattr(openai, "Completion", None)
            if chat_completion:
                response = chat_completion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            elif completion:
                response = completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=1500,
                    temperature=0.7,
                )
                return response.choices[0].text.strip()
            else:
                return mock_constitution()
        except Exception as e:
            return f"[OpenAI Error: {e}]\n\nFalling back to mock constitution.\n" + mock_constitution()
    else:
        return mock_constitution()

def mock_constitution():
    return (
        "Article 1: All citizens have equal rights.\n"
        "Article 2: Legislature is elected every 4 years.\n"
        "Article 3: Judiciary is independent.\n"
        "Article 4: Universal Basic Income is guaranteed.\n"
        "Article 5: Freedom of press, speech, and association.\n"
        "Article 6: Citizens may recall politicians by referendum.\n"
        "Article 7: Elections are proportional.\n"
        "Article 8: No supreme court; judicial review by panels.\n"
        "... (etc)"
    )

# --- Section 2: Streamlit UI Layout ---
st.set_page_config(page_title="AI Constitution Simulator", layout="wide")
st.title("🧠💡 AI Constitution Simulator (Modular, PhD-level)")

with st.expander("ℹ️ About this app", expanded=False):
    st.write("""
    - **Generate**: Use an LLM (OpenAI) or mock generator to create a constitution.
    - **Simulate**: Run a multi-agent society with citizens, politicians, judges, journalists, and media.
    - **Analyze**: Visualize metrics: inequality, unrest, happiness, polarization, corruption, press freedom, protests, and more.
    - **Compare & Optimize**: Run and rank multiple constitutions, evolve them to maximize outcomes.
    - **Import Scenarios**: Seed the simulation with real-world data.
    """)

# --- Sidebar: Controls ---
st.sidebar.header("Simulation Controls")
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
prompt = st.sidebar.text_area("Constitution Prompt", value="Draft a 5-page constitution that maximizes democratic participation, economic equality, and judicial independence.")
N_citizens = st.sidebar.slider("Number of citizens", 50, 1000, 200, step=10)
N_politicians = st.sidebar.slider("Number of politicians", 1, 20, 5)
N_judges = st.sidebar.slider("Number of judges", 1, 10, 3)
N_journalists = st.sidebar.slider("Number of journalists", 1, 10, 3)
years = st.sidebar.slider("Years to simulate", 5, 100, 30)
random_seed = st.sidebar.number_input("Random seed (0=random)", 0, 999999, 0)

# --- Tabs for Dashboard Expansion ---
tabs = st.tabs([
    "Simulation",
    "Agent Logs",
    "Event Playback",
    "Optimization",
    "Scenario Import",
    "Advanced Analytics",
    "Network Animation & Communities",
    "Constitution Comparison",
    "Leaderboard",
])

# --- Tab 1: Simulation ---
with tabs[0]:
    st.header("1️⃣ Generate & Simulate Society")
    if st.button("Generate Constitution"):
        with st.spinner("Generating constitution..."):
            constitution = generate_constitution(prompt, api_key if api_key else None)
        st.subheader("Generated Constitution")
        st.code(constitution)
        st.session_state['constitution'] = constitution
    elif 'constitution' in st.session_state:
        constitution = st.session_state['constitution']
        st.subheader("Generated Constitution")
        st.code(constitution)
    else:
        constitution = mock_constitution()
        st.session_state['constitution'] = constitution
        st.subheader("Generated Constitution (default)")
        st.code(constitution)

    # Parse constitution
    rules = parse_constitution_text(constitution)
    st.markdown(f"**Extracted Rules:** {rules}")

    if st.button("Run Simulation"):
        with st.spinner("Simulating society..."):
            model = SocietyModel(
                N_citizens=N_citizens,
                N_politicians=N_politicians,
                N_judges=N_judges,
                N_journalists=N_journalists,
                constitution_text=constitution,
                constitution_params=rules,
                years=years,
                seed=random_seed if random_seed != 0 else None
            )
            model.run()
            metrics_df = model.get_metrics_df()
            st.session_state['metrics_df'] = metrics_df
            st.session_state['model'] = model
    else:
        metrics_df = st.session_state.get('metrics_df', None)
        model = st.session_state.get('model', None)

    if metrics_df is not None:
        st.subheader("Outcome Metrics Over Time")
        st.dataframe(metrics_df.tail())
        # --- Plots ---
        st.markdown("### 📊 Key Metrics")
        fig, axs = plt.subplots(2, 3, figsize=(18, 8))
        sns.lineplot(data=metrics_df, x='year', y='gini', ax=axs[0,0])
        axs[0,0].set_title('Inequality (Gini)')
        sns.lineplot(data=metrics_df, x='year', y='avg_dissent', ax=axs[0,1])
        axs[0,1].set_title('Unrest (Avg Dissent)')
        sns.lineplot(data=metrics_df, x='year', y='avg_happiness', ax=axs[0,2])
        axs[0,2].set_title('Happiness')
        sns.lineplot(data=metrics_df, x='year', y='polarization', ax=axs[1,0])
        axs[1,0].set_title('Ideology Polarization')
        sns.lineplot(data=metrics_df, x='year', y='avg_corruption', ax=axs[1,1])
        axs[1,1].set_title('Corruption')
        sns.lineplot(data=metrics_df, x='year', y='press_freedom', ax=axs[1,2])
        axs[1,2].set_title('Press Freedom')
        plt.tight_layout()
        st.pyplot(fig)
        # --- Protests and Laws Overturned ---
        st.markdown("### 🪧 Protests & Judicial Review")
        fig2, ax2 = plt.subplots(1,2,figsize=(12,4))
        sns.barplot(x='year', y='protests', data=metrics_df, ax=ax2[0])
        ax2[0].set_title('Protests (1=Yes, 0=No)')
        sns.barplot(x='year', y='laws_overturned', data=metrics_df, ax=ax2[1])
        ax2[1].set_title('Laws Overturned')
        plt.tight_layout()
        st.pyplot(fig2)
        # --- Social Network Visualization ---
        st.markdown("### 🕸️ Social Network Snapshot")
        if model:
            G = nx.Graph()
            for c in model.citizens:
                G.add_node(c.id, type='citizen', ideology=c.ideology)
                for f in c.social_links:
                    G.add_edge(c.id, f)
            pos = nx.spring_layout(G, seed=42)
            fig3, ax3 = plt.subplots(figsize=(8,6))
            node_list = list(G.nodes)
            node_colors = [float(model.agents[n].ideology) if n in model.agents else 0.0 for n in node_list]
            cmap = plt.get_cmap('coolwarm')
            if len(node_colors) == len(node_list):
                import numpy as np
                node_colors_arr = np.array(node_colors, dtype=float)
                nx.draw_networkx_nodes(
                    G, pos, nodelist=node_list, node_color=node_colors_arr, cmap=cmap, ax=ax3, node_size=100, vmin=-1, vmax=1  # type: ignore
                )
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color='blue', ax=ax3)
            nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax3)
            ax3.set_title('Citizen Social Network (color=ideology)')
            ax3.axis('off')
            st.pyplot(fig3)
        # --- Summary ---
        st.markdown("### 📝 Simulation Summary")
        st.write(f"**Years simulated:** {years}")
        st.write(f"**Final Gini (Inequality):** {metrics_df['gini'].iloc[-1]:.3f}")
        st.write(f"**Final Avg Dissent (Unrest):** {metrics_df['avg_dissent'].iloc[-1]:.3f}")
        st.write(f"**Final Avg Happiness:** {metrics_df['avg_happiness'].iloc[-1]:.3f}")
        st.write(f"**Final Polarization:** {metrics_df['polarization'].iloc[-1]:.3f}")
        st.write(f"**Final Corruption:** {metrics_df['avg_corruption'].iloc[-1]:.3f}")
        st.write(f"**Final Press Freedom:** {metrics_df['press_freedom'].iloc[-1]:.3f}")
        st.write(f"**Total Protests:** {metrics_df['protests'].sum()}")
        st.write(f"**Laws Overturned:** {metrics_df['laws_overturned'].iloc[-1]}")
        # --- Export ---
        st.markdown("### 📤 Export Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Export CSV"):
                export_csv(metrics_df, "simulation_results.csv")
                st.success("Exported to simulation_results.csv")
        with col2:
            if st.button("Export JSON"):
                export_json(metrics_df, "simulation_results.json")
                st.success("Exported to simulation_results.json")
        with col3:
            if st.button("Export PDF"):
                try:
                    export_pdf(metrics_df, "simulation_results.pdf")
                    st.success("Exported to simulation_results.pdf")
                except ImportError:
                    st.error("reportlab is not installed.")
        with col4:
            if st.button("Generate Styled PDF Report"):
                from analytics import generate_pdf_report
                try:
                    pdf_path = generate_pdf_report(model, metrics_df, filename="simulation_report.pdf")
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF Report", f, file_name="simulation_report.pdf")
                except ImportError:
                    st.error("reportlab is not installed.")

# --- Tab 2: Agent Logs ---
with tabs[1]:
    st.header("2️⃣ Agent Logs & Timelines")
    model = st.session_state.get('model', None)
    if model:
        agent_ids = [a.id for a in model.citizens + model.politicians + model.judges + model.journalists]
        selected_id = st.selectbox("Select Agent ID", agent_ids)
        agent = model.agents[selected_id]
        st.write(f"**Type:** {agent.type}")
        st.write(f"**Role:** {agent.role}")
        st.write(f"**Log:**")
        st.code("\n".join(agent.log[-20:]))
        st.write(f"**Memory:**")
        st.json(agent.memory[-10:])
        st.write(f"**Timeline (Happiness):**")
        happiness_timeline = [float(log.split('happiness=')[1].split(',')[0]) if 'happiness=' in log else 0 for log in agent.log if 'happiness=' in log]
        if happiness_timeline:
            fig, ax = plt.subplots()
            plot_agent_timeline(happiness_timeline, ax=ax)
            st.pyplot(fig)

# --- Tab 3: Event Playback ---
with tabs[2]:
    st.header("3️⃣ Event Playback & History")
    model = st.session_state.get('model', None)
    if model:
        st.write("**Event Log:**")
        st.code("\n".join(model.event_log[-50:]))
        st.write("**Step-by-step Playback:**")
        if 'playback_step' not in st.session_state:
            st.session_state['playback_step'] = 1
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("⏪ Rewind"):
            st.session_state['playback_step'] = 1
        if col2.button("◀️ Back"):
            st.session_state['playback_step'] = max(1, st.session_state['playback_step'] - 1)
        if col3.button("▶️ Forward"):
            st.session_state['playback_step'] = min(model.year, st.session_state['playback_step'] + 1)
        if col4.button("⏩ Fast Forward"):
            st.session_state['playback_step'] = model.year
        step = st.session_state['playback_step']
        st.write(f"**Metrics at Year {step}:**")
        metrics_df = model.get_metrics_df()
        st.dataframe(metrics_df[metrics_df['year'] == step])
        st.markdown("---")
        st.markdown("### ⚡ What-if Scenario Engine")
        event_type = st.selectbox("Inject Event", ["revolution", "new_law", "economic_boom", "pandemic"])
        law = None
        if event_type == "new_law":
            law = st.text_input("Law Name", value="Universal Healthcare")
        if st.button("Inject Event"):
            if event_type == "new_law":
                model.inject_event(event_type, law=law)
            else:
                model.inject_event(event_type)
            st.success(f"Event '{event_type}' injected!")

# --- Tab 4: Optimization ---
with tabs[3]:
    st.header("4️⃣ Constitution Optimization")
    metric = st.selectbox("Metric to Optimize", ["gini", "avg_happiness", "avg_dissent", "press_freedom"])
    prompt_template = st.text_area("Prompt Template", value=prompt)
    n_generations = st.slider("Generations", 1, 20, 5)
    if st.button("Run Optimizer"):
        with st.spinner("Optimizing constitutions..."):
            history = optimize_constitution(metric, prompt_template, n_generations, generate_constitution, api_key if api_key else "")
        st.write("**Best Constitutions by Generation:**")
        for i, entry in enumerate(history):
            st.markdown(f"**Gen {i+1}:** {entry['prompt']}")
            st.code(entry['constitution'])
            st.write(f"{metric}: {entry[metric]:.3f}")

# --- Tab 5: Scenario Import ---
with tabs[4]:
    st.header("5️⃣ Scenario Import (Real-World Data)")
    st.info("Upload a CSV with columns: wealth, ideology, happiness, dissent, etc. to seed the simulation.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        st.success("Scenario imported!")
        model = st.session_state.get('model', None)
        if model:
            if st.button("Apply Scenario to Current Simulation"):
                model.seed_from_dataframe(df)
                st.success("Scenario applied to current simulation!")
        else:
            if st.button("Run Simulation with Scenario"):
                # Use default parameters, but seed with uploaded data
                model = SocietyModel(
                    N_citizens=len(df),
                    N_politicians=N_politicians,
                    N_judges=N_judges,
                    N_journalists=N_journalists,
                    constitution_text=st.session_state.get('constitution', mock_constitution()),
                    years=years,
                    seed=random_seed if random_seed != 0 else None
                )
                model.seed_from_dataframe(df)
                model.run()
                metrics_df = model.get_metrics_df()
                st.session_state['metrics_df'] = metrics_df
                st.session_state['model'] = model
                st.success("Simulation run with scenario data!")

# --- Tab 6: Advanced Analytics ---
with tabs[5]:
    st.header("6️⃣ Advanced Analytics & Trade-offs")
    metrics_df = st.session_state.get('metrics_df', None)
    if metrics_df is not None:
        st.markdown("### 📉 Trade-off Plot: Freedom vs. Stability vs. Equality")
        fig, ax = plt.subplots(figsize=(8,6))
        plot_tradeoff(metrics_df, 'gini', 'press_freedom', 'avg_dissent', ax=ax)
        st.pyplot(fig)

# --- Tab 7: Network Animation & Communities ---
with tabs[6]:
    st.header("7️⃣ Network Animation & Community Detection")
    model = st.session_state.get('model', None)
    metrics_df = st.session_state.get('metrics_df', None)
    if model and metrics_df is not None:
        st.markdown("#### Animate Social Network Evolution")
        if st.button("Generate Network Animation (GIF)"):
            from analytics import animate_network_evolution
            gif_path = animate_network_evolution(model.history, model, filename="network_evolution.gif")
            with open(gif_path, "rb") as f:
                st.image(f.read())
        st.markdown("#### Community Detection & Visualization")
        from analytics import plot_communities
        fig, ax = plt.subplots(figsize=(8,6))
        plot_communities(model, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Run a simulation first to enable network animation and community detection.")

# --- Tab 8: Constitution Comparison ---
with tabs[7]:
    st.header("8️⃣ Constitution Comparison Tool")
    from constitution_parser import compare_constitutions
    st.markdown("Paste two constitutions below to compare their structure and extracted rules.")
    text1 = st.text_area("Constitution 1", value=mock_constitution(), height=200)
    text2 = st.text_area("Constitution 2", value=mock_constitution(), height=200)
    if st.button("Compare Constitutions"):
        diff = compare_constitutions(text1, text2)
        if diff:
            st.write("**Differences:**")
            st.json(diff)
        else:
            st.success("No differences found in extracted rules.")

# --- Tab 9: Leaderboard ---
with tabs[8]:
    st.header("9️⃣ Leaderboard: Best Constitutions by Metric")
    if 'leaderboard' not in st.session_state:
        st.session_state['leaderboard'] = []
    # Add current simulation to leaderboard
    metrics_df = st.session_state.get('metrics_df', None)
    model = st.session_state.get('model', None)
    if metrics_df is not None and model is not None:
        if st.button("Add Current Simulation to Leaderboard"):
            entry = {
                'constitution': st.session_state.get('constitution', ''),
                'metrics': metrics_df.iloc[-1].to_dict(),
            }
            st.session_state['leaderboard'].append(entry)
            st.success("Added to leaderboard!")
    # Display leaderboard
    if st.session_state['leaderboard']:
        st.markdown("### 🏆 Leaderboard Table")
        leaderboard_df = pd.DataFrame([{'Gini': e['metrics']['gini'],
                                        'Happiness': e['metrics']['avg_happiness'],
                                        'Dissent': e['metrics']['avg_dissent'],
                                        'Press Freedom': e['metrics']['press_freedom'],
                                        'Constitution': e['constitution'][:60] + '...'}
                                       for e in st.session_state['leaderboard']])
        st.dataframe(leaderboard_df)
        st.download_button("Download Leaderboard (CSV)", leaderboard_df.to_csv(index=False), file_name="leaderboard.csv")
        # Save/load/share
        if st.button("Save Leaderboard to File"):
            leaderboard_df.to_csv("leaderboard_saved.csv", index=False)
            st.success("Leaderboard saved to leaderboard_saved.csv")
        if st.button("Load Leaderboard from File"):
            try:
                loaded_df = pd.read_csv("leaderboard_saved.csv")
                for _, row in loaded_df.iterrows():
                    st.session_state['leaderboard'].append({'constitution': row['Constitution'],
                                                            'metrics': {'gini': row['Gini'],
                                                                        'avg_happiness': row['Happiness'],
                                                                        'avg_dissent': row['Dissent'],
                                                                        'press_freedom': row['Press Freedom']}})
                st.success("Leaderboard loaded from leaderboard_saved.csv")
            except Exception as e:
                st.error(f"Failed to load leaderboard: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
:rocket: **AI Constitution Simulator** | Modular, extensible, and ready for research or fun.\
[GitHub](https://github.com/) | [MIT License](LICENSE)
""") 