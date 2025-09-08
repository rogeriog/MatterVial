# Create placeholder README
echo "# SHAP Plots

This directory contains SHAP visualization plots for MatterVial features.

The plots are automatically downloaded from Figshare when first accessed through the Interpreter class.

- **Archive size**: ~100MB (compressed)
- **Extracted size**: ~531MB
- **Total files**: 3,638 SVG files
- **Figshare URL**: [Your Figshare URL]

## Manual Download

If you prefer to download manually:

\`\`\`bash
cd mattervial/interpreter/
wget [YOUR_FIGSHARE_URL] -O shap_plots_archive.tar.gz
tar -xzf shap_plots_archive.tar.gz
\`\`\`
" > mattervial/interpreter/shap_plots/README.md
