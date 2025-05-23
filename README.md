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

## Bussiner Understanding

