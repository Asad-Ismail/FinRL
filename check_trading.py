# Check Portfolio Performance
# ## Get cumulative return
df_erl, cumu_erl = alpaca_history(
    key=DATA_API_KEY,
    secret=DATA_API_SECRET,
    url=DATA_API_BASE_URL,
    start="2023-05-20",  # must be within 1 month
    end="2023-05-30",
)  # change the date if error occurs

df_djia, cumu_djia = DIA_history(start="2023-05-20")
returns_erl = cumu_erl - 1
returns_dia = cumu_djia - 1
returns_dia = returns_dia[: returns_erl.shape[0]]

# plot and save
import matplotlib.pyplot as plt

plt.figure(dpi=1000)
plt.grid()
plt.grid(which="minor", axis="y")
plt.title("Stock Trading (Paper trading)", fontsize=20)
plt.plot(returns_erl, label="ElegantRL Agent", color="red")
# plt.plot(returns_sb3, label = 'Stable-Baselines3 Agent', color = 'blue' )
# plt.plot(returns_rllib, label = 'RLlib Agent', color = 'green')
plt.plot(returns_dia, label="DJIA", color="grey")
plt.ylabel("Return", fontsize=16)
plt.xlabel("Year 2021", fontsize=16)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(78))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
#ax.xaxis.set_major_formatter(
#    ticker.FixedFormatter(["", "10-19", "", "10-20", "", "10-21", "", "10-22"])
plt.legend(fontsize=10.5)
plt.savefig("papertrading_stock.png")
