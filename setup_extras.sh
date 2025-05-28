#!/usr/bin/env bash
set -e

# Create .external directory if it doesn't exist
mkdir -p .external
cd .external

# === Clone and setup GetContacts ===
if [ ! -d getcontacts ]; then
    echo "📦 Cloning GetContacts..."
    git clone https://github.com/getcontacts/getcontacts
else
    echo "✅ GetContacts already cloned."
fi

# Add GetContacts to PATH for this session
export PATH="$(pwd)/getcontacts:$PATH"
echo "🔗 GetContacts added to PATH for this session."

# === Clone and install py-mfdca ===
if [ ! -d py-mfdca ]; then
    echo "📦 Cloning py-mfdca..."
    git clone https://github.com/utdal/py-mfdca.git
else
    echo "✅ py-mfdca already cloned."
fi

cd py-mfdca
echo "🔧 Installing py-mfdca..."
pip install .
cd ../..

echo "✅ All extra packages installed successfully."
