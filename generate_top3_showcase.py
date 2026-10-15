import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

def build_showcase():
    fig = plt.figure(figsize=(25, 15), facecolor='#0d1117', dpi=120)
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.09, 0.91], hspace=0.10, wspace=0.08,
                           left=0.03, right=0.97, top=0.96, bottom=0.03)

    # Title Header
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis('off')
    ax_head.text(0.5, 0.68, 'ALDEBARAN EXOPLANET SURVEY — TOP-3 PLANET CANDIDATES',
                 ha='center', va='center', color='#58a6ff', fontsize=26, fontweight='bold', family='sans-serif')
    ax_head.text(0.5, 0.18, 'NASA ExoFOP CTOI Shortlist | Status: Unconfirmed Planet Candidates Awaiting High-Res Imaging & RV Follow-Up',
                 ha='center', va='center', color='#8b949e', fontsize=14, family='sans-serif')

    cards = [
        {
            'tid': 301152169,
            'title': 'TIC 301152169 (Sub-Neptune Candidate)',
            'subtitle': 'Periodic Transit Signal around Bright Pulsating A/F Host',
            'badge': 'CANDIDATE: SUB-NEPTUNE',
            'badge_col': '#d97706',
            'dossier': 'tess_results/deep_vetting/dossier_TIC_301152169.png',
            'stats': [
                ('Fitted Radius (Rp)', '4.50 R_Earth (Candidate)', '#58a6ff'),
                ('Signal Period', '3.103757 days (~74.5 h)', '#ffffff'),
                ('Transit Depth', '707.6 ppm', '#ffffff'),
                ('Transit Duration', '2.59 hours', '#ffffff'),
                ('Detection SNR', '28.64 sigma (5 modes removed)', '#3fb950'),
                ('Host Star', 'V=9.22 mag, Teff=7797 K (A/F)', '#ffffff'),
                ('Observed Data', '29 transits (Sectors 100-103)', '#ffffff'),
                ('Current Status', 'Unconfirmed Candidate (CTOI tier)', '#f0883e'),
                ('Required Follow-Up', 'Spectroscopy RV (K~5 m/s) & Speckle', '#bc8cff')
            ]
        },
        {
            'tid': 350190949,
            'title': 'TIC 350190949 (USP Earth Candidate)',
            'subtitle': 'Periodic Dips with 7.8-Year Multi-Sector Baseline',
            'badge': 'CANDIDATE: USP EARTH (22.2h)',
            'badge_col': '#238636',
            'dossier': 'tess_results/deep_vetting/dossier_TIC_350190949.png',
            'stats': [
                ('Fitted Radius (Rp)', '1.08 R_Earth (Candidate)', '#58a6ff'),
                ('Signal Period', '0.926120 days (22.2 hours)', '#ffffff'),
                ('Transit Depth', '1346.9 ppm', '#ffffff'),
                ('Transit Duration', '0.81 hours (49 minutes)', '#ffffff'),
                ('Observed Transits', '94 transits across 7.8-year span', '#3fb950'),
                ('Host Star', 'M-dwarf (0.31 R_Sun, Teff=3361 K)', '#ffffff'),
                ('Multi-Sector Baseline', 'Sectors 1 (2018), 28, 104, 105 (2026)', '#ffffff'),
                ('Current Status', 'Unconfirmed Candidate (CTOI tier)', '#f0883e'),
                ('Required Follow-Up', 'High-res imaging & ESPRESSO RV', '#bc8cff')
            ]
        },
        {
            'tid': 322051865,
            'title': 'TIC 322051865 (Sub-Earth Candidate)',
            'subtitle': 'Periodic Dips in Low-Contamination Field (0 NEB)',
            'badge': 'CANDIDATE: SUB-EARTH (0.88 R_E)',
            'badge_col': '#1f6feb',
            'dossier': 'tess_results/deep_vetting/dossier_TIC_322051865.png',
            'stats': [
                ('Fitted Radius (Rp)', '0.88 R_Earth (Candidate)', '#58a6ff'),
                ('Signal Period', '2.831532 days (~68.0 h)', '#ffffff'),
                ('Transit Depth', '778.4 ppm', '#ffffff'),
                ('Transit Duration', '1.08 hours', '#ffffff'),
                ('Equilibrium Temp', 'T_eq ~ 538 K (if confirmed)', '#3fb950'),
                ('Host Star', 'M-dwarf (0.33 R_Sun, Teff=3368 K)', '#ffffff'),
                ('Observed Data', '31 transits across Sectors 103-106', '#ffffff'),
                ('Current Status', 'Unconfirmed Candidate (CTOI tier)', '#f0883e'),
                ('Required Follow-Up', 'Bayesian FPP calculation & RV', '#bc8cff')
            ]
        }
    ]

    for col_idx, card in enumerate(cards):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            spine.set_linewidth(1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Badge pill
        badge_text = f"  {card['badge']}  "
        ax.text(0.5, 0.965, badge_text, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold', family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=card['badge_col'], edgecolor='none'))

        # Card Title
        ax.text(0.5, 0.925, card['title'], ha='center', va='center',
                color='#f0f6fc', fontsize=15, fontweight='bold', family='sans-serif')
        ax.text(0.5, 0.895, card['subtitle'], ha='center', va='center',
                color='#8b949e', fontsize=11, family='sans-serif')

        # Direct crop from full dossier [0,0] quadrant
        dos_im = Image.open(card['dossier'])
        im_crop = dos_im.crop((20, 25, 1180, 765))

        ax_img = ax.inset_axes([0.03, 0.44, 0.94, 0.43])
        ax_img.imshow(im_crop)
        ax_img.axis('off')

        # Stats table
        y_start = 0.40
        dy = 0.042
        for s_idx, (label, val, val_col) in enumerate(card['stats']):
            cur_y = y_start - s_idx * dy
            ax.text(0.06, cur_y, label + ':', ha='left', va='center',
                    color='#8b949e', fontsize=11, fontweight='normal', family='sans-serif')
            ax.text(0.94, cur_y, val, ha='right', va='center',
                    color=val_col, fontsize=11, fontweight='bold', family='sans-serif')
            if s_idx < len(card['stats']) - 1:
                ax.plot([0.06, 0.94], [cur_y - dy/2, cur_y - dy/2], color='#21262d', lw=0.8)

    out_path = 'tess_results/exofop_top3_showcase.png'
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f'Successfully updated {out_path} with strict candidate labeling!')

if __name__ == '__main__':
    build_showcase()
