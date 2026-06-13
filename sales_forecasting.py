import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def generate_sales_data():
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    t = np.arange(n)

    trend = 500 + 0.18 * t

    # weekday effect: Mon=0 through Sun=6
    weekly = np.array([0, -20, -10, 10, 30, 80, 60])
    week_effect = np.array([weekly[d.weekday()] for d in dates])

    annual = (
        60 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
        + 120 * np.where((pd.DatetimeIndex(dates).month == 11) | (pd.DatetimeIndex(dates).month == 12), 1, 0)
        - 40 * np.where(pd.DatetimeIndex(dates).month.isin([6, 7, 8]), 1, 0)
    )

    promo = np.zeros(n)
    for date in dates:
        idx = (dates == date).argmax()
        if date.month == 11 and 24 <= date.day <= 30:
            promo[idx] = 250
        if date.month == 2 and 12 <= date.day <= 16:
            promo[idx] = 100
        if date.month == 8 and date.day >= 20:
            promo[idx] = 80

    noise = np.random.normal(0, 40, n)
    sales = np.maximum(trend + week_effect + annual + promo + noise, 50).round(2)

    return pd.DataFrame({'date': dates, 'sales': sales})


FEATURE_COLS = [
    'dayofweek', 'month', 'quarter', 'dayofmonth', 'dayofyear',
    'weekofyear', 'year', 'is_weekend', 't_index',
    'sin_year_1', 'cos_year_1', 'sin_year_2', 'cos_year_2',
    'sin_year_3', 'cos_year_3', 'sin_week', 'cos_week',
    'lag_7', 'lag_14', 'lag_21', 'lag_28',
    'roll_mean_7', 'roll_mean_14', 'roll_mean_30',
    'roll_std_7', 'roll_std_14', 'roll_std_30',
    'is_november', 'is_december', 'is_summer',
]


def add_features(df, drop_na=True):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['dayofmonth'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['t_index'] = np.arange(len(df))

    for k in [1, 2, 3]:
        df[f'sin_year_{k}'] = np.sin(2 * np.pi * k * df['dayofyear'] / 365)
        df[f'cos_year_{k}'] = np.cos(2 * np.pi * k * df['dayofyear'] / 365)
    df['sin_week'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_week'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    for lag in [7, 14, 21, 28]:
        df[f'lag_{lag}'] = df['sales'].shift(lag)

    for win in [7, 14, 30]:
        df[f'roll_mean_{win}'] = df['sales'].shift(1).rolling(win).mean()
        df[f'roll_std_{win}'] = df['sales'].shift(1).rolling(win).std()

    df['is_november'] = (df['month'] == 11).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

    if drop_na:
        df = df.dropna().reset_index(drop=True)
    return df


def train_models(df):
    cutoff = df['date'].max() - pd.Timedelta(days=90)
    train = df[df['date'] <= cutoff].copy()
    test = df[df['date'] > cutoff].copy()

    X_train = train[FEATURE_COLS]
    y_train = train['sales']
    X_test = test[FEATURE_COLS]
    y_test = test['sales']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name == 'Ridge Regression':
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results[name] = {
            'model': model, 'scaler': scaler, 'preds': preds,
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape,
            'use_scaler': (name == 'Ridge Regression')
        }
        print(f"  {name:25s}  MAE={mae:6.1f}  RMSE={rmse:6.1f}  R²={r2:.4f}  MAPE={mape:.1f}%")

    return results, train, test, scaler


def forecast_future(df, best_name, results, scaler, n_days=60):
    best = results[best_name]
    model = best['model']
    use_scaler = best['use_scaler']

    last_date = df['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days, freq='D')

    future_stub = pd.DataFrame({'date': future_dates, 'sales': np.nan})
    extended = pd.concat([df[['date', 'sales']], future_stub], ignore_index=True)
    ext_feat = add_features(extended, drop_na=False)
    future_feat = ext_feat[ext_feat['date'].isin(future_dates)]

    predictions = []
    rolling_sales = list(df['sales'].values[-60:])

    for _, row in future_feat.iterrows():
        row_copy = row.copy()

        for lag in [7, 14, 21, 28]:
            row_copy[f'lag_{lag}'] = rolling_sales[-lag] if len(rolling_sales) >= lag else np.mean(rolling_sales)

        for win in [7, 14, 30]:
            window = rolling_sales[-win:] if len(rolling_sales) >= win else rolling_sales
            row_copy[f'roll_mean_{win}'] = np.mean(window)
            row_copy[f'roll_std_{win}'] = np.std(window)

        X_row = pd.DataFrame([row_copy[FEATURE_COLS]])
        if use_scaler:
            X_row = scaler.transform(X_row)

        pred = max(model.predict(X_row)[0], 50)
        predictions.append(pred)
        rolling_sales.append(pred)

    out = pd.DataFrame({'date': future_dates, 'forecast': predictions})
    recent_std = df['sales'].tail(90).std()
    out['lower'] = (out['forecast'] - 1.5 * recent_std).clip(lower=0)
    out['upper'] = out['forecast'] + 1.5 * recent_std
    return out


COLORS = {
    'blue': '#1B6CA8',
    'orange': '#F4A261',
    'green': '#2E9E6B',
    'red': '#E63946',
    'bg': '#F8F9FA',
    'grid': '#E0E0E0',
    'dark': '#1A1A2E',
    'mid': '#555570',
}


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(COLORS['bg'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['mid'], labelsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.6, linestyle='--')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['dark'], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=COLORS['mid'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=COLORS['mid'])


def plot_dashboard(df, results, future_df, best_name, train, test):
    fig = plt.figure(figsize=(20, 26), facecolor='white')
    fig.suptitle('Retail Sales Forecasting Dashboard', fontsize=20, fontweight='bold',
                 color=COLORS['dark'], y=0.98)

    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.32,
                           top=0.94, bottom=0.04, left=0.07, right=0.97)

    # --- historical sales + rolling trend ---
    ax1 = fig.add_subplot(gs[0, :])
    monthly = df.groupby(pd.Grouper(key='date', freq='ME'))['sales'].sum().reset_index()
    monthly['trend'] = monthly['sales'].rolling(3).mean()

    ax1.fill_between(monthly['date'], monthly['sales'], alpha=0.25, color=COLORS['blue'])
    ax1.plot(monthly['date'], monthly['sales'], color=COLORS['blue'], linewidth=2.2, label='Monthly Sales')
    ax1.plot(monthly['date'], monthly['trend'], color=COLORS['orange'], linewidth=2.5,
             linestyle='--', label='3-Month Trend')

    for yr in [2022, 2023, 2024]:
        ax1.axvspan(pd.Timestamp(f'{yr}-11-01'), pd.Timestamp(f'{yr}-12-31'),
                    alpha=0.10, color=COLORS['orange'])
    ax1.text(pd.Timestamp('2022-11-15'), monthly['sales'].max() * 0.95,
             'Holiday Seasons', fontsize=8, color=COLORS['orange'], ha='center')

    style_ax(ax1, '3-Year Monthly Sales History & Trend', ylabel='Total Sales ($)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.legend(fontsize=10, framealpha=0.9)

    # --- day of week ---
    ax2 = fig.add_subplot(gs[1, 0])
    dow = df.groupby('dayofweek')['sales'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    bar_colors = [COLORS['green'] if i >= 5 else COLORS['blue'] for i in range(7)]
    bars = ax2.bar(days, dow.values, color=bar_colors, edgecolor='white', linewidth=0.8, zorder=3)
    for bar, val in zip(bars, dow.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'${val:.0f}', ha='center', va='bottom', fontsize=8.5,
                 fontweight='bold', color=COLORS['dark'])
    style_ax(ax2, 'Avg Daily Sales by Day of Week', ylabel='Avg Sales ($)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # --- month of year ---
    ax3 = fig.add_subplot(gs[1, 1])
    mon = df.groupby('month')['sales'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors_m = [COLORS['red'] if m in [11, 12] else (COLORS['green'] if m in [6, 7, 8] else COLORS['blue'])
                for m in range(1, 13)]
    bars_m = ax3.bar(month_names, mon.values, color=colors_m, edgecolor='white', linewidth=0.8, zorder=3)
    for bar, val in zip(bars_m, mon.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'${val:.0f}', ha='center', va='bottom', fontsize=7.5,
                 fontweight='bold', color=COLORS['dark'])
    style_ax(ax3, 'Avg Daily Sales by Month', ylabel='Avg Sales ($)')
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax3.tick_params(axis='x', labelsize=8)
    ax3.legend(handles=[
        Patch(color=COLORS['red'], label='Peak (Nov-Dec)'),
        Patch(color=COLORS['green'], label='Summer Dip'),
        Patch(color=COLORS['blue'], label='Normal'),
    ], fontsize=8, loc='upper left', framealpha=0.9)

    # --- model error comparison ---
    ax4 = fig.add_subplot(gs[2, 0])
    model_names = list(results.keys())
    maes = [results[m]['MAE'] for m in model_names]
    rmses = [results[m]['RMSE'] for m in model_names]
    x = np.arange(len(model_names))
    w = 0.35
    b1 = ax4.bar(x - w / 2, maes, w, label='MAE', color=COLORS['blue'], zorder=3)
    b2 = ax4.bar(x + w / 2, rmses, w, label='RMSE', color=COLORS['orange'], zorder=3)
    for bar in list(b1) + list(b2):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'${bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color=COLORS['dark'])
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    style_ax(ax4, 'Model Error Comparison (lower = better)', ylabel='Error ($)')
    ax4.legend(fontsize=9)

    # --- scorecard table ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(COLORS['bg'])
    ax5.axis('off')
    ax5.set_title('Model Scorecard', fontsize=12, fontweight='bold', color=COLORS['dark'], pad=12)

    headers = ['Model', 'R²', 'MAPE', 'MAE ($)', 'Best?']
    rows = []
    for name in model_names:
        r = results[name]
        rows.append([name, f"{r['R2']:.4f}", f"{r['MAPE']:.1f}%", f"${r['MAE']:.1f}",
                     'Yes' if name == best_name else ''])

    tbl = ax5.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                    bbox=[0, 0.1, 1, 0.85])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    best_r2 = max(results[m]['R2'] for m in model_names)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(COLORS['grid'])
        if row == 0:
            cell.set_facecolor(COLORS['blue'])
            cell.set_text_props(color='white', fontweight='bold')
        elif results[model_names[row - 1]]['R2'] == best_r2:
            cell.set_facecolor('#D4EDDA')
        else:
            cell.set_facecolor('white' if row % 2 == 0 else '#F5F5F5')

    # --- actual vs predicted ---
    ax6 = fig.add_subplot(gs[3, :])
    y_test = test['sales']
    y_pred = results[best_name]['preds']

    ax6.plot(test['date'], y_test.values, color=COLORS['mid'], linewidth=1.4, alpha=0.8, label='Actual')
    ax6.plot(test['date'], y_pred, color=COLORS['red'], linewidth=1.8,
             linestyle='--', label=f'Predicted ({best_name})')
    ax6.fill_between(test['date'], y_test.values, y_pred, alpha=0.12, color=COLORS['red'], label='Error')

    style_ax(ax6, f'Actual vs Predicted — Test Period (Last 90 Days)', xlabel='Date', ylabel='Daily Sales ($)')
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax6.legend(fontsize=10, framealpha=0.9, loc='upper left')

    # --- 60-day forecast ---
    ax7 = fig.add_subplot(gs[4, :])
    hist_tail = df.tail(60)
    ax7.plot(hist_tail['date'], hist_tail['sales'], color=COLORS['blue'], linewidth=1.6, label='Historical')
    ax7.plot(future_df['date'], future_df['forecast'], color=COLORS['green'], linewidth=2.2,
             linestyle='--', label='60-Day Forecast')
    ax7.fill_between(future_df['date'], future_df['lower'], future_df['upper'],
                     alpha=0.18, color=COLORS['green'], label='Confidence Band (±1.5σ)')

    divider = df['date'].max()
    ax7.axvline(divider, color=COLORS['orange'], linewidth=2, linestyle=':')
    ax7.text(divider, ax7.get_ylim()[1] * 0.97, '  Historical | Forecast  ',
             fontsize=9, color=COLORS['orange'], va='top', ha='center')

    max_idx = future_df['forecast'].idxmax()
    ax7.annotate(
        f"Peak: ${future_df.loc[max_idx, 'forecast']:,.0f}",
        xy=(future_df.loc[max_idx, 'date'], future_df.loc[max_idx, 'forecast']),
        xytext=(future_df.loc[max_idx, 'date'] - pd.Timedelta(days=10),
                future_df.loc[max_idx, 'forecast'] + 60),
        fontsize=9, fontweight='bold', color=COLORS['green'],
        arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5)
    )

    style_ax(ax7, '60-Day Sales Forecast with Confidence Band', xlabel='Date', ylabel='Daily Sales ($)')
    ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax7.legend(fontsize=10, framealpha=0.9)

    plt.savefig('/mnt/user-data/outputs/sales_forecasting_dashboard.png', dpi=160,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Dashboard saved.")


def plot_feature_importance(results, best_name):
    model = results[best_name]['model']
    if not hasattr(model, 'feature_importances_'):
        print(f"Note: {best_name} doesn't expose feature importances, skipping chart.")
        return

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor(COLORS['bg'])

    bar_cols = [COLORS['blue'] if i < 5 else COLORS['orange'] if i < 10 else COLORS['green']
                for i in range(len(imp))]
    bars = ax.barh(imp.index[::-1], imp.values[::-1], color=bar_cols[::-1], edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, imp.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8.5, color=COLORS['dark'])

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['mid'], labelsize=9)
    ax.set_xlabel('Importance Score', fontsize=10, color=COLORS['mid'])
    ax.set_title(f'Top 15 Predictive Features ({best_name})', fontsize=13,
                 fontweight='bold', color=COLORS['dark'], pad=12)
    ax.grid(axis='x', color=COLORS['grid'], linewidth=0.6, linestyle='--')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Feature importance chart saved.")


def export_data(future_df, results):
    future_df.to_csv('/mnt/user-data/outputs/forecast_data.csv', index=False)

    metrics = [{'Model': name, 'MAE': round(r['MAE'], 2), 'RMSE': round(r['RMSE'], 2),
                'R2': round(r['R2'], 4), 'MAPE': round(r['MAPE'], 2)}
               for name, r in results.items()]
    pd.DataFrame(metrics).to_csv('/mnt/user-data/outputs/model_metrics.csv', index=False)
    print("CSVs saved.")


def print_summary(results, future_df, best_name, df):
    r = results[best_name]
    total = future_df['forecast'].sum()
    avg = future_df['forecast'].mean()
    peak_date = future_df.loc[future_df['forecast'].idxmax(), 'date']
    low_date = future_df.loc[future_df['forecast'].idxmin(), 'date']

    same_period_last_year = df[df['date'].between(
        future_df['date'].min() - pd.DateOffset(years=1),
        future_df['date'].max() - pd.DateOffset(years=1)
    )]['sales'].sum()
    yoy = ((total - same_period_last_year) / same_period_last_year * 100) if same_period_last_year else 0

    print(f"""
=== EXECUTIVE SUMMARY ===

Best Model: {best_name}
  R²:   {r['R2']:.4f}
  MAPE: {r['MAPE']:.1f}%
  MAE:  ${r['MAE']:.0f}/day

60-Day Forecast ({future_df['date'].min().date()} to {future_df['date'].max().date()})
  Total projected sales: ${total:,.0f}
  Daily average:         ${avg:,.0f}
  Peak day:              {peak_date.date()} (${future_df['forecast'].max():,.0f})
  Slowest day:           {low_date.date()} (${future_df['forecast'].min():,.0f})
  YoY change:            {yoy:+.1f}%

Business Recommendations
  Inventory: buffer ~15-20% above daily average (${avg * 1.18:,.0f}/day)
  Staffing:  add weekend shifts, scale back Mon-Tue
  Cash flow: ~${total * 0.12:,.0f} reserve recommended for next procurement cycle
  Confidence band: ±${future_df['upper'].sub(future_df['forecast']).mean():,.0f}/day
""")


if __name__ == '__main__':
    print("Generating sales data...")
    df_raw = generate_sales_data()

    print("Adding features...")
    df = add_features(df_raw)

    print("Training models...")
    results, train, test, scaler = train_models(df)

    best_name = max(results, key=lambda m: results[m]['R2'])
    print(f"\nBest model: {best_name} (R²={results[best_name]['R2']:.4f})\n")

    print("Generating 60-day forecast...")
    future_df = forecast_future(df, best_name, results, scaler)

    print("Plotting...")
    plot_dashboard(df, results, future_df, best_name, train, test)
    plot_feature_importance(results, best_name)
    export_data(future_df, results)

    print_summary(results, future_df, best_name, df)
