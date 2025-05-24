# Laporan Proyek Machine Learning - Mochammad Syafiq Ilallah

## Project Overview

## Later Belakang
Dalam era transformasi digital, sektor ritel menghadapi tantangan besar dalam meramalkan permintaan konsumen yang semakin dinamis. Ketidakpastian dalam perilaku pembelian, fluktuasi harga minyak, serta pengaruh eksternal seperti hari libur dan promosi, membuat estimasi penjualan menjadi semakin kompleks. Kemampuan untuk memprediksi penjualan secara akurat sangat penting untuk mengoptimalkan rantai pasokan, manajemen stok, hingga strategi pemasaran yang tepat sasaran (Sajawal et al., 2022).

Metode tradisional seperti regresi linier atau model statistik sederhana sering kali gagal menangkap pola non-linier dan interaksi antar fitur yang kompleks. Oleh karena itu, pendekatan berbasis machine learning menjadi semakin relevan dalam dunia ritel modern. Teknik seperti Random Forest dan XGBoost telah terbukti mampu meningkatkan akurasi prediksi penjualan dengan menangani data berskala besar, bersifat multivariat, serta mengandung noise (Mustapha & Sithole, 2025; Swami et al., 2020).

Dalam konteks ini, digunakan dataset Store Sales - Time Series Forecasting dari Kaggle yang menyediakan data penjualan harian dari berbagai toko di Ekuador. Dataset ini mencakup informasi penting seperti tanggal, jenis produk, promosi, lokasi toko, transaksi, harga minyak dunia, dan data hari libur. Kompleksitas dan kelengkapan dataset tersebut memungkinkan eksplorasi menyeluruh terhadap faktor-faktor yang memengaruhi penjualan.

Penelitian ini bertujuan untuk membangun model prediksi penjualan ritel berbasis Random Forest dan XGBoost, dengan terlebih dahulu melalui tahapan preprocessing data seperti penanganan data hilang, encoding fitur kategorikal, normalisasi, serta deteksi dan pembersihan outlier. Proses ini penting agar model dapat belajar dari data historis dengan lebih representatif dan menghasilkan prediksi yang dapat diandalkan dalam praktik nyata. Dengan model yang akurat dan stabil, bisnis ritel dapat membuat keputusan yang lebih strategis dan berbasis data.

### Mengapa dan bagaimana masalah tersebut harus diselesaikan
Permasalahan prediksi penjualan menjadi sangat penting bagi bisnis ritel seperti yang tercermin dalam dataset Store Sales - Time Series Forecasting dari Kaggle, yang berisi data penjualan harian dari berbagai toko di Ekuador. Dataset ini mencakup variabel kompleks seperti jenis produk (family), promosi (onpromotion), transaksi harian, lokasi toko, hari libur, serta harga minyak dunia. Ketika variabel-variabel ini saling berinteraksi, prediksi penjualan tidak lagi dapat dilakukan dengan metode sederhana karena pola penjualan menjadi sangat fluktuatif dan dipengaruhi banyak faktor eksternal. Tanpa prediksi yang akurat, perusahaan berisiko mengalami kelebihan stok, kekurangan suplai, hingga kerugian finansial. Oleh karena itu, diperlukan pendekatan berbasis machine learning untuk menghasilkan model prediktif yang dapat menangani kompleksitas tersebut secara efektif (Sajawal et al., 2022; Mustapha & Sithole, 2025).

Untuk mengatasi tantangan tersebut, proyek ini membangun model prediksi penjualan menggunakan algoritma Random Forest dan XGBoost berdasarkan data dari Kaggle. Data terlebih dahulu diproses melalui tahapan seperti menggabungkan beberapa sumber data (penjualan, toko, transaksi, minyak, hari libur), menangani nilai hilang, encoding fitur kategorikal, normalisasi numerik, serta pembersihan outlier. Model kemudian dilatih untuk memprediksi nilai penjualan (sales) dan dievaluasi dengan metrik MSE, MAE, dan R². Hasil evaluasi menunjukkan bahwa XGBoost memiliki performa yang lebih stabil dibandingkan Random Forest pada data uji, yang menandakan kemampuannya dalam melakukan generalisasi terhadap data baru (Swami et al., 2020). Dengan pendekatan ini, prediksi penjualan yang dihasilkan dapat digunakan untuk mendukung keputusan operasional toko secara lebih akurat dan adaptif terhadap dinamika pasar.

### Referensi
Sajawal, M., Usman, S., Alshaikh, H. S., Hayat, A., & Ashraf, M. U. (2022). A predictive analysis of retail sales forecasting using machine learning techniques. Lahore Garrison University Research Journal of Computer Science and Information Technology, 6(4). https://doi.org/10.54692/lgurjcsit.2022.0604399

Mustapha, O. O., & Sithole, T. (2025). Forecasting retail sales using machine learning models. American Journal of Statistical and Actuarial Science, 6(1), 35–67. https://doi.org/10.47672/ajsas.2679

Swami, D., Shah, A. D., & Ray, S. K. B. (2020). Predicting future sales of retail products using machine learning. arXiv preprint. https://arxiv.org/abs/2008.07779

## **Business Understanding**

Permasalahan utama yang terdapat dalam industri ritel adalah kesulitan dalam memprediksi penjualan secara akurat karena banyaknya variabel yang saling memengaruhi secara kompleks, seperti jenis produk, promosi, lokasi toko, jumlah transaksi, hari libur, dan harga minyak. Perusahaan yang tidak mampu memprediksi penjualan dengan baik berisiko mengalami kelebihan stok, kekurangan suplai, atau kerugian finansial. Model tradisional seringkali gagal dalam menangkap pola ini, sehingga diperlukan pendekatan berbasis *machine learning* untuk meningkatkan akurasi prediksi.

Dalam konteks ini, dalam membangun model prediksi penjualan ritel dapat menggunakan algoritma seperti **Random Forest** dan **XGBoost**, dengan tahapan yang komprehensif seperti penggabungan multi-sumber data, pembersihan data, encoding kategori, normalisasi fitur numerik, serta deteksi dan penanganan outlier.

### **Problem Statement**

Bagaimana membangun model prediksi penjualan ritel harian yang akurat dan stabil dengan memanfaatkan data multivariat dari berbagai toko di Ekuador menggunakan algoritma machine learning seperti Random Forest dan XGBoost?

### **Goals**

Menghasilkan model prediksi penjualan berbasis machine learning yang mampu menangkap hubungan kompleks antar fitur dan mampu melakukan generalisasi dengan baik terhadap data baru, sehingga dapat digunakan sebagai dasar pengambilan keputusan operasional dan strategis toko ritel yang mengarah pada data.

### **Solution Statements**

1. **Menggunakan dua algoritma machine learning** yaitu Random Forest dan XGBoost untuk membangun dan membandingkan performa model prediksi penjualan.
2. **Melakukan data preprocessing menyeluruh**, termasuk penggabungan data dari berbagai file (`train.csv`, `stores.csv`, `transactions.csv`, `oil.csv`, `holidays_events.csv`), penanganan missing value dengan interpolasi dan imputation berbasis grup, serta encoding fitur kategorikal dan normalisasi numerik.
3. **Melakukan evaluasi performa model** menggunakan metrik terukur seperti Mean Squared Error (MSE), Mean Absolute Error (MAE), dan R² Score pada data training dan testing.
4. **Meningkatkan akurasi dan stabilitas model** melalui pemilihan model dengan generalisasi terbaik berdasarkan evaluasi — dalam hal ini XGBoost dipilih karena memberikan keseimbangan performa pada data latih dan uji.

## **Data Understanding**

Dataset yang digunakan dalam proyek ini adalah **Store Sales - Time Series Forecasting** yang tersedia di [Kaggle Datasets](https://www.kaggle.com/datasets/shiyonisagar/store-sales-time-series-forecasting). Dataset ini merupakan data penjualan harian dari berbagai toko di Ekuador dan sangat cocok digunakan untuk proyek peramalan penjualan karena mencakup berbagai fitur yang memengaruhi performa toko, seperti promosi, hari libur, transaksi, lokasi toko, dan faktor eksternal seperti harga minyak dunia.

Dataset ini terdiri dari beberapa file terpisah yang masing-masing memiliki peran penting dalam membentuk konteks yang lebih lengkap untuk analisis dan pemodelan prediksi penjualan.

### **Variabel-variabel dalam dataset ini adalah sebagai berikut:**

#### **1. train.csv**

| Fitur         | Deskripsi                                                        |
| ------------- | ---------------------------------------------------------------- |
| id          | ID unik untuk setiap observasi.                                  |
| date        | Tanggal dalam format YYYY-MM-DD.                                 |
| store_nbr   | Nomor toko tempat produk dijual.                                 |
| family      | Kategori produk, seperti BEVERAGES, BREAD/BAKERY, CLEANING, dll. |
| sales       | Jumlah penjualan unit per hari. (Target variabel)                |
| onpromotion | Jumlah item dalam kategori tersebut yang sedang dalam promosi.   |


#### **2. stores.csv**

| Fitur       | Deskripsi                                                          |
| ----------- | ------------------------------------------------------------------ |
| store_nbr | Nomor toko.                                                        |
| city      | Kota tempat toko berada.                                           |
| state     | Provinsi atau negara bagian di Ekuador.                            |
| type      | Tipe toko (A–E), menggambarkan format atau ukuran toko.            |
| cluster   | Segmentasi toko berdasarkan karakteristik demografis dan perilaku. |


#### **3. transactions.csv**

| Fitur          | Deskripsi                                       |
| -------------- | ----------------------------------------------- |
| date         | Tanggal transaksi.                              |
| store_nbr    | Nomor toko.                                     |
| transactions | Jumlah transaksi yang terjadi di toko tersebut. |


#### **4. oil.csv**

| Fitur        | Deskripsi                   |
| ------------ | --------------------------- |
| date       | Tanggal pengukuran.         |
| dcoilwtico | Harga minyak WTI dalam USD. |


#### **5. holidays\_events.csv**

| Fitur         | Deskripsi                                                             |
| ------------- | --------------------------------------------------------------------- |
| date        | Tanggal event.                                                        |
| type        | Jenis event (Holiday, Transfer, Bridge, Additional, Event, Work Day). |
| locale      | Tingkat cakupan (Local, Regional, National).                          |
| locale_name | Nama kota atau negara bagian tempat berlaku.                          |
| description | Nama atau deskripsi hari libur/event.                                 |
| transferred | Boolean yang menunjukkan apakah event dipindah tanggalnya.            |



### **Tahapan Tambahan: Exploratory Data Analysis (EDA)**

Untuk memahami struktur dan karakteristik data secara lebih mendalam, dilakukan beberapa langkah EDA:

* Visualisasi distribusi kategori toko (type) dan wilayah (state) menunjukkan bahwa toko tipe D dan wilayah Pichincha paling dominan jumlahnya, yang bisa mempengaruhi agregat penjualan nasional.
* Distribusi jumlah transaksi cenderung menceng ke kanan, dengan puncak pada kisaran 1000–1500 transaksi.
* Harga minyak (dcoilwtico) menunjukkan pola bimodal, dengan dua puncak harga dominan di sekitar \$50 dan \$100.
* Distribusi hari libur menunjukkan dominasi hari libur nasional dibanding lokal atau regional.

EDA dan pemahaman variabel ini sangat penting agar model machine learning dapat dibangun di atas fondasi data yang bersih, representatif, dan bermakna.


