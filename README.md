# Defensive Impact Score (DIS) 🏀

Welcome to the **Defensive Impact Score (DIS)**!  
DIS is my own custom stat that brings a new way to look at NBA defense, going beyond the box score to capture how much impact players (and teams) really have on that end of the floor.  

The DIS-App is the way I decided to share it. Built with **Python + Streamlit**, this app makes it easy to explore, compare, and visualize defensive value in a fun and interactive way.

---

## ✨ What You Can Do
- **Player Profiles** → Check a player’s DIS history and compare them with others.  
- **Team Profiles** → See how whole teams perform, with weighted minutes and rankings.  
- **Leaderboards** → Browse player and team leaderboards for every season.  
- **Filters** → Apply minutes or games-played filters so only serious contributors show up.  

---

## 📸 Screenshots

### Player Profile Example
![Alex Caruso Player Profile Example](https://github.com/user-attachments/assets/5463cf67-6be8-47dc-942c-afa4a990c15e)

### Team Profile Example
![Utah Jazz Team Profile Example](https://github.com/user-attachments/assets/fdf27540-9b21-417e-b9ef-b7171ecf0198)

### Player Leaderboard Page Example
![2024-2025 Player Leaderboard Example](https://github.com/user-attachments/assets/984a9c82-9d22-443b-a0a7-581f14d18544)

---

## ⚙️ How to Run It Yourself
1. Clone this repo:
   ```bash
   git clone https://github.com/magioria/DIS-App.git
   cd DIS-App
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run DIS_app.py
   ```

4. Open the link (usually `http://localhost:8501`) and start exploring! 🚀  

---

## 📂 What’s Inside
```
DIS-App/
│── outputs/          # Cleaned CSV outputs
│── DIS_app.py        # Main Streamlit app (navigation & pages)
│── data_utils.py     # Data loading and season ordering
│── styling.py        # Table styling, category colors, and percentiles
│── plots.py          # Matplotlib/Streamlit charts and visualizations
│── profiles.py       # Player and team profile rendering
│── leaderboard.py    # Leaderboard pagination and rendering
│── requirements.txt  # Python dependencies
└── README.md         # You’re here!
```

---

## 🔮 What’s Coming Next
- Add **correlation tools** for deeper analysis.  
- Add a **forecasting mode** to predict defensive trends.  

---

## 🙌 Why I Built This
Defense is the hardest part of basketball to measure. Box score stats don’t tell the full story, so I turned to **advanced stats, z-scores and custom-built defensive models** to build something that captures value more realistically.  

The goal: make defense measurable, comparable, and fun to explore.

---

### 📑 Data Sources
This project uses publicly available basketball data from [Basketball Reference](https://www.basketball-reference.com/), [NBA.com/stats](https://www.nba.com/stats), and [BBall Index](https://www.bball-index.com/).  
All data is used strictly for **educational, research, and non-commercial purposes** in the creation of the DIS metric.  

---

## 📜 License & Professional Use
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  

- ✅ Free to use for personal, educational, or research purposes.  
- ✅ You must credit **Matteo Capucci** (author).  
- ❌ Commercial use is not allowed without explicit permission.  

If you’re seriously interested in my work, whether it’s for **collaboration or professional projects**, feel free to reach out:  

👉 Created by **Matteo Capucci** ([@magioria](https://github.com/magioria))  
📩 [Connect on LinkedIn](https://www.linkedin.com/in/matteo-capucci/)  

See the [LICENSE](LICENSE) file for full details.  
