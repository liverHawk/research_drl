echo "####### No.1......"
apt install -y -V ca-certificates lsb-release wget
echo "####### No.2......"
wget https://packages.apache.org/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
echo "####### No.3......"
apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
echo "####### No.4......"
apt update
echo "####### No.5......"
apt install -y -V libarrow-dev # For C++
echo "####### No.6......"
apt install -y -V libarrow-glib-dev # For GLib (C)
echo "####### No.7......"
apt install -y -V libarrow-dataset-dev # For Apache Arrow Dataset C++
echo "####### No.8......"
apt install -y -V libarrow-dataset-glib-dev # For Apache Arrow Dataset GLib (C)
echo "####### No.9......"
apt install -y -V libarrow-acero-dev # For Apache Arrow Acero
echo "####### No.10....."
apt install -y -V libarrow-flight-dev # For Apache Arrow Flight C++
echo "####### No.11....."
apt install -y -V libarrow-flight-glib-dev # For Apache Arrow Flight GLib (C)
echo "####### No.12....."
apt install -y -V libarrow-flight-sql-dev # For Apache Arrow Flight SQL C++
echo "####### No.13....."
apt install -y -V libarrow-flight-sql-glib-dev # For Apache Arrow Flight SQL GLib (C)
echo "####### No.14....."
apt install -y -V libgandiva-dev # For Gandiva C++
echo "####### No.15....."
apt install -y -V libgandiva-glib-dev # For Gandiva GLib (C)
echo "####### No.16....."
apt install -y -V libparquet-dev # For Apache Parquet C++
echo "####### No.17....."
apt install -y -V libparquet-glib-dev # For Apache Parquet GLib (C)

rm -f apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb