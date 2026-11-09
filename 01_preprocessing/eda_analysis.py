import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, 'raw_data', 'Bank Marketing.csv')
IMAGES_DIR = os.path.join(CURRENT_DIR, 'images')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_data(file_path: str) -> pd.DataFrame:
    """Завантажує вхідний датасет."""
    if not os.path.exists(file_path):
        print(f"Помилка: файл не знайдено: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path)
    print(f"Дані успішно завантажено ({df.shape[0]} рядків).")
    return df


def plot_class_imbalance(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 1: Дисбаланс цільової змінної Subscription (88/12)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    target_counts = df['Subscription'].value_counts()
    labels = ['Відмова (Клас 1)', 'Згода (Клас 2)']
    values = [target_counts.get(1, 0), target_counts.get(2, 0)]
    colors = ['#2b5c8f', '#e26d5c']

    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor='black', linewidth=1)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    ax.set_title('Дисбаланс класів цільової змінної Subscription (88.3% проти 11.7%)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Кількість спостережень (клієнтів)', fontsize=11)
    ax.set_ylim(0, max(values) * 1.18)

    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_class_imbalance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено графік 1: {output_path}")
    return output_path


def plot_numerical_distributions(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 2: Розподіли основних числових колонок (фокус на 97–99% клієнтів: Медіана vs Середнє)."""
    configs = [
        {
            'col': 'Age',
            'title': 'Вік клієнта (Age)',
            'unit': 'р.',
            'xlim': (18, 75),
            'bins': 28,
            'kde': True,
            'discrete': False,
            'val_fmt': '.1f',
            'tail_note': 'Максимум: 95 років\n(99.4% клієнтів віком до 75 р.;\nрозподіл близький до нормального)'
        },
        {
            'col': 'Balance (euros)',
            'title': 'Баланс на рахунку (Balance)',
            'unit': '€',
            'xlim': (-2000, 7500),
            'bins': 45,
            'kde': True,
            'discrete': False,
            'val_fmt': '.0f',
            'tail_note': 'Хвіст до 102 127 € (10.5% аномалій)\n(багаті вкладники зсувають\nсереднє далеко вправо від медіани!)'
        },
        {
            'col': 'Last Contact Duration',
            'title': 'Тривалість останньої розмови (Duration)',
            'unit': 'с',
            'xlim': (0, 1200),
            'bins': 40,
            'kde': True,
            'discrete': False,
            'val_fmt': '.0f',
            'tail_note': 'Хвіст до 4 918 с (~82 хв)\n(98.8% дзвінків до 20 хв;\nдовгі розмови ведуть до успіху)'
        },
        {
            'col': 'Campaign',
            'title': 'Кількість контактів у кампанії (Campaign)',
            'unit': 'дзв.',
            'xlim': (0.3, 10.7),
            'bins': 10,
            'kde': False,
            'discrete': True,
            'val_fmt': '.1f',
            'tail_note': 'Хвіст до 63 дзвінків\n(97.4% клієнтів мають ≤ 10 дзвінків;\nмедіана = 2 контакти)'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=300)
    axes = axes.flatten()

    for i, cfg in enumerate(configs):
        ax = axes[i]
        col = cfg['col']

        data_in_range = df[(df[col] >= cfg['xlim'][0]) & (df[col] <= cfg['xlim'][1])][col]

        if cfg['discrete']:
            sns.histplot(
                data_in_range, discrete=True, ax=ax,
                color='#1d3557', stat='count', alpha=0.55, edgecolor='#1d3557',
                shrink=0.85
            )
            ax.set_xticks(range(1, 11))
        else:
            sns.histplot(
                data_in_range, kde=cfg['kde'], ax=ax,
                color='#1d3557', bins=cfg['bins'], stat='count', alpha=0.55, edgecolor='#1d3557'
            )

        mean_val = df[col].mean()
        median_val = df[col].median()

        fmt = cfg['val_fmt']
        dur_extra_med = f' (~{median_val/60:.1f} хв)' if col == 'Last Contact Duration' else ''
        dur_extra_mean = f' (~{mean_val/60:.1f} хв)' if col == 'Last Contact Duration' else ''

        ax.axvline(median_val, color='#e63946', linestyle='--', linewidth=2.5,
                   label=f'Медіана: {median_val:{fmt}} {cfg["unit"]}{dur_extra_med}')
        ax.axvline(mean_val, color='#2a9d8f', linestyle=':', linewidth=2.5,
                   label=f'Середнє: {mean_val:{fmt}} {cfg["unit"]}{dur_extra_mean}')

        ax.text(
            0.97, 0.70, cfg['tail_note'],
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3b0', edgecolor='#e09f3e', alpha=0.9)
        )

        ax.set_xlim(cfg['xlim'])
        ax.set_title(cfg['title'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(f"{col} ({cfg['unit']})", fontsize=10, fontweight='bold')
        ax.set_ylabel('Кількість клієнтів', fontsize=10)
        ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

    plt.suptitle('Розподіли основних числових ознак (Фокус на 97–99% вибірки: Медіана vs Середнє)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '02_numerical_distributions.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено зрозумілий графік 2: {output_path}")
    return output_path


def plot_outliers_iqr(df: pd.DataFrame, output_dir: str) -> str:
    """Слайд 3: Діаграми розмаху (Boxplot) та правило 1.5 IQR (людино-читабельний вигляд)."""
    configs = [
        {
            'col': 'Age',
            'title': 'Вік клієнта (Age)',
            'unit': 'років',
            'xlim': (10, 95),
            'clamp_min': 18,
            'tail_note': 'Максимум: 95 років\n(пенсіонери, не помилка)'
        },
        {
            'col': 'Balance (euros)',
            'title': 'Баланс на рахунку (Balance)',
            'unit': '€',
            'xlim': (-2500, 7500),
            'clamp_min': None,
            'tail_note': 'Хвіст до 102 127 €\n(VIP-вкладники, 10.5% вибірки)'
        },
        {
            'col': 'Last Contact Duration',
            'title': 'Тривалість розмови (Duration)',
            'unit': 'с',
            'xlim': (0, 1100),
            'clamp_min': 0,
            'tail_note': 'Хвіст до 4 918 с (~82 хв)\n(тривалі результативні розмови)'
        },
        {
            'col': 'Campaign',
            'title': 'Кількість дзвінків (Campaign)',
            'unit': 'дзв.',
            'xlim': (0, 14),
            'clamp_min': 1,
            'tail_note': 'Максимум: до 63 дзвінків\n(інтенсивна кампанія)'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=300)
    axes = axes.flatten()

    for i, cfg in enumerate(configs):
        ax = axes[i]
        col = cfg['col']
        s = df[col]

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if cfg['clamp_min'] is not None and lower < cfg['clamp_min']:
            display_lower = None
        else:
            display_lower = lower

        outliers = ((s < lower) | (s > upper)).sum()
        pct = (outliers / len(s)) * 100

        ax.boxplot(
            s, orientation='horizontal', patch_artist=True, widths=0.45,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='#e63946', markeredgecolor='none', markersize=4, alpha=0.35),
            medianprops=dict(color='#1d3557', linewidth=2.5),
            boxprops=dict(facecolor='#a8dadc', edgecolor='#1d3557', linewidth=1.5),
            whiskerprops=dict(color='#1d3557', linewidth=1.5),
            capprops=dict(color='#1d3557', linewidth=1.5)
        )

        ax.axvspan(cfg['xlim'][0], upper, color='#2a9d8f', alpha=0.18, label='Зона норми (до 1.5 IQR)')
        ax.axvspan(upper, cfg['xlim'][1], color='#e63946', alpha=0.18, label=f'Зона аномалій ({pct:.1f}%)')
        ax.axvline(upper, color='#b7094c', linestyle='--', linewidth=2, label=f'Поріг 1.5 IQR: {upper:.0f} {cfg["unit"]}')

        if display_lower is not None:
            ax.axvline(display_lower, color='#b7094c', linestyle=':', linewidth=1.5, label=f'Нижня межа: {display_lower:.0f} {cfg["unit"]}')

        med = s.median()
        ax.text(med, 1.28, f'Медіана: {med:.0f}', ha='center', fontsize=9, fontweight='bold', color='#1d3557')

        ax.text(
            0.97, 0.22, cfg['tail_note'],
            transform=ax.transAxes, ha='right', va='center',
            fontsize=9, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3b0', edgecolor='#e09f3e', alpha=0.9)
        )

        ax.set_xlim(cfg['xlim'])
        ax.set_yticks([])
        ax.set_xlabel(f"{col} ({cfg['unit']})", fontsize=10, fontweight='bold')
        ax.set_title(f"{cfg['title']} | Викидів за 1.5 IQR: {outliers:,} ({pct:.1f}%)", fontsize=11, fontweight='bold', pad=10)
        ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

    plt.suptitle('Виявлення аномалій за правилом 1.5 IQR (Фокусний масштаб на розподілі більшості клієнтів)', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_outliers_iqr_boxplots.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Створено зрозумілий графік 3: {output_path}")
    return output_path


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    df = load_data(INPUT_FILE)

    print("Генерація 3 графіків для презентації...")
    plot_class_imbalance(df, IMAGES_DIR)
    plot_numerical_distributions(df, IMAGES_DIR)
    plot_outliers_iqr(df, IMAGES_DIR)

    print(f"Усі 3 графіки успішно збережено в: {IMAGES_DIR}")


if __name__ == '__main__':
    main()
