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
| dcoilwtico (oil_price) | Harga minyak WTI dalam USD. |


#### **5. holidays\_events.csv**

| Fitur         | Deskripsi                                                             |
| ------------- | --------------------------------------------------------------------- |
| date        | Tanggal event.                                                        |
| type (holiday_type)       | Jenis event (Holiday, Transfer, Bridge, Additional, Event, Work Day). |
| locale (holiday_locale)     | Tingkat cakupan (Local, Regional, National).                          |
| locale_name (holiday_locale_name) | Nama kota atau negara bagian tempat berlaku.                          |
| description (holiday_desc) | Nama atau deskripsi hari libur/event.                                 |
| transferred | Boolean yang menunjukkan apakah event dipindah tanggalnya.            |



### **Tahapan Tambahan: Exploratory Data Analysis (EDA)**

Untuk memahami struktur dan karakteristik data secara lebih mendalam, dilakukan beberapa langkah EDA:

* Visualisasi distribusi kategori toko (type) dan wilayah (state) menunjukkan bahwa toko tipe D dan wilayah Pichincha paling dominan jumlahnya, yang bisa mempengaruhi agregat penjualan nasional.
* Distribusi jumlah transaksi cenderung menceng ke kanan, dengan puncak pada kisaran 1000–1500 transaksi.
* Harga minyak (dcoilwtico) menunjukkan pola bimodal, dengan dua puncak harga dominan di sekitar \$50 dan \$100.
* Distribusi hari libur menunjukkan dominasi hari libur nasional dibanding lokal atau regional.

EDA dan pemahaman variabel ini sangat penting agar model machine learning dapat dibangun di atas fondasi data yang bersih, representatif, dan bermakna.

## **Data Preparation**

Pada bagian ini dilakukan serangkaian tahapan preprocessing untuk memastikan kualitas data yang digunakan dalam proses pelatihan model prediksi penjualan. Berikut adalah langkah-langkah data preparation secara berurutan:
1. Merge data + *Rename* data + Penambahan Fitur
2. Penanganan missing value
   - Menggunakan interpolate(), ffill(), dan bfill() digunakan untuk mengisi nilai hilang pada oil_price.
   - Menggunakan nilai `transactions` diisi berdasarkan toko menggunakan groupby + ffill() dan bfill().
   - Kategori hari libur yang kosong diisi dengan nilai default ("NoHoliday", "None").
   - Penanganan Outlier
3. Encoding kategori fitur dengan One-Hot-Encoding
4. Normalisasi data dengan MinMaxScaler
5. Sampling dan Pembagian Data (Train-Test Split)
6. Konversi Tipe Data Numerik

**Rubik Tambahan**

### **1.  Merge data + *Rename* data + Penambahan Fitur**

**Teknik**:
* Data dari berbagai file (`train.csv`, `stores.csv`, `transactions.csv`, `oil.csv`, dan `holidays_events.csv`) digabungkan menggunakan `pd.merge()` berdasarkan kolom `store_nbr` dan `date`.
* Kolom dcoilwtico dari `oil.csv` di-*rename* menjadi oil_price dengan .rename() untuk meningkatkan keterbacaan dan konsistensi.
* Kolom type, locale, locale_name, dan description dari `holidays_events.csv` diubah menjadi holiday_type, holiday_locale, holiday_locale_name, dan holiday_desc dengan menggunakan .rename() untuk  menghindari konflik saat penggabungan.
* Ditambahkan fitur biner baru bernama `real_holiday` dengan memanfaatkan fungsi def determine_real_holiday untuk menyatakan apakah suatu tanggal merupakan hari libur yang relevan bagi toko tersebut. Fitur ini dibuat berdasarkan kombinasi lokasi libur (National, Regional, Local) dan atribut lokasi toko (state, city).

**Tujuan**:
* Menyatukan informasi penting ke dalam satu dataset untuk pelatihan model, serta menghindari konflik nama kolom dan meningkatkan keterbacaan.
* Penambahan fitur `real_holiday` menyederhanakan representasi kompleks hari libur ke dalam bentuk numerik yang bisa langsung dimanfaatkan model prediktif.

### **2. Menangani Nilai Hilang (Missing Values)**

**Teknik**:
* interpolate(method='linear') digunakan untuk mengisi nilai kosong pada kolom `oil_price` secara linier antar waktu.
* ffill() dan bfill() melengkapi sisa missing value di awal atau akhir data.
* groupby("store_nbr")["transactions"].transform(...) memastikan imputasi transactions dilakukan per toko agar sesuai pola tiap toko dan menjaga indeks tetap utuh.
* fillna() digunakan untuk mengganti nilai kosong pada kolom kategori hari libur dengan label default seperti "NoHoliday" atau "None" agar tidak dianggap sebagai missing value saat analisis atau modeling.
* Menggunakan visualisasi boxplot untuk mendeteksi outlier.
* Menghapus nilai ekstrem dengan metode IQR (Interquartile Range).

**Tujuan**:
* Untuk menangani missing value pada oil_price dan transaction itu menggunakan teknik di atas bertujuan untuk menghindari error pada saat pelatihan model dan memastikan data tetap representatif secara historis.
* Untuk outlier bertujuan untuk meningkatkan kualitas data dan menghindari pengaruh ekstrem yang dapat mendistorsi prediksi model.

### **3.Encoding kategori fitur dengan One-Hot-Encoding**

**Teknik**:
* Fitur kategorikal seperti family, type, city, holiday_type, dll. dikonversi menjadi kolom biner (0/1) menggunakan pd.get_dummies() dan menambahakan kolom hasil encoding ke data frame dan hapus kolom kategori lama dengan kolom kategori yang sudah di encoding.

**Tujuan**:
* Menjadikan data kategorikal dapat diproses oleh model machine learning, yang umumnya hanya menerima data bersifat numerik.

### **4. Normalisasi data dengan MinMaxScaler**

**Teknik**:
* Menggunakan `MinMaxScaler` pada kolom numerik seperti onpromotion, cluster, oil_price, transactions, real_holiday.

**Tujuan**:
* Menyamakan skala antar fitur agar model tidak bias terhadap fitur dengan rentang nilai yang besar.

### **5. Sampling dan Pembagian Data (Train-Test Split)**

**Teknik**:
* Diambil sampel acak sebanyak 30.000 baris dari data bersih untuk efisiensi komputasi.
* Data dibagi menjadi fitur (X) dan target (y), lalu dilakukan split 80% train dan 20% test menggunakan `train_test_split`.

**Tujuan**:
* Mengurangi beban komputasi dan memastikan evaluasi model dilakukan pada data yang belum dilatih.

### **6. Konversi Tipe Data Numerik**

**Teknik**:
* Kolom numerik seperti sales, transactions, oil_price, onpromotion, cluster, dan real_holiday dikonversi dari tipe `float` ke `int` menggunakan .astype(int).

**Tujuan**:
* Menyesuaikan format tipe data dengan kebutuhan analisis lanjutan dan interpretasi (misalnya ketika ingin menampilkan dalam bentuk tabel ringkas).
* Menghindari masalah kompatibilitas saat menyimpan, visualisasi, atau inverse-scaling nilai prediksi.
* Pada konteks seperti onpromotion dan transactions, data bersifat diskret sehingga lebih logis direpresentasikan dalam bentuk integer.

## MODELING

### 1. **Random Forest Regressor**

Random Forest adalah algoritma **ensemble learning** berbasis pohon keputusan (decision tree). Ia membangun banyak pohon secara paralel dari subset acak data dan fitur, lalu menggabungkan hasilnya. Untuk regresi, prediksi akhirnya adalah rata-rata dari semua pohon.

Menurut Mustapha & Sithole (2025), algoritma ini sangat sesuai untuk prediksi penjualan karena mampu menangani data berskala besar, fitur multivariat, dan noise dengan baik, serta tahan terhadap overfitting.

**Parameter yang Digunakan**:
* n_estimators=100 : Jumlah pohon keputusan yang dibangun.
* random_state=42 : Nilai acak untuk memastikan hasil pelatihan model konsisten setiap kali dijalankan.
* n_jobs=-1 : Menggunakan seluruh inti CPU untuk mempercepat proses pelatihan.

**Cara Kerja**:
* Dataset dilatih dengan metode **bootstrap sampling** (pengambilan subset acak dengan pengembalian).
* Setiap pohon dilatih dengan subset acak dari fitur untuk meningkatkan keragaman antar pohon.
* Prediksi akhir dihitung sebagai **rata-rata** dari prediksi semua pohon.

**Keunggulan**:

* Mampu menangkap hubungan non-linier antar fitur.
* Tahan terhadap overfitting karena efek rata-rata dari banyak pohon.
* Mudah digunakan dan relatif stabil terhadap data noisy.

### 2. **XGBoost Regressor**

XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang bekerja secara bertahap untuk mengoreksi kesalahan dari model sebelumnya. Cocok untuk data yang kompleks dan besar, serta dikenal karena kecepatan dan akurasinya yang tinggi.

Menurut Sajawal et al. (2022) dan Swami et al. (2020), XGBoost sangat efektif dalam kasus prediksi penjualan karena dapat menangani missing value, data tidak seimbang, dan memiliki kemampuan regularisasi untuk mencegah overfitting.

**Parameter yang Digunakan**:

* n_estimators=300 : Jumlah pohon boosting yang dibangun secara bertahap.
* max_depth=4 : Maksimal kedalaman pohon untuk mengontrol kompleksitas model.
* random_state=42 : Seed acak untuk memastikan hasil yang dapat direproduksi.
* n_jobs=-1 : Menggunakan seluruh inti CPU untuk efisiensi proses pelatihan.

**Cara Kerja**:

* Pohon pertama dibuat untuk menghasilkan prediksi awal.
* Setiap pohon berikutnya dibentuk untuk meminimalkan error dari pohon sebelumnya, menggunakan **gradient descent** terhadap loss function.
* Prediksi akhir diperoleh dari penjumlahan semua pohon yang dibangun.

**Keunggulan**:

* Akurat dan efisien, sangat cocok untuk data time series dan multivariat.
* Mendukung regularisasi (L1 dan L2), sehingga lebih tahan terhadap overfitting.
* Memiliki fitur otomatis untuk penanganan missing value dan optimalisasi paralel.


## EVALUATION

Dalam evaluasi model prediksi penjualan harian, beberapa metrik evaluasi digunakan untuk mengukur performa model yang dibangun. Metrik ini dipilih karena relevansinya dalam konteks regresi (prediksi nilai kontinu), serta kemampuannya memberi wawasan berbeda terkait kualitas prediksi.

### **Metrik yang Digunakan**:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R-squared (R²)

### **Hasil Evaluasi Model**
Berdasarkan hasil evaluasi pada data train dan test, peforma kedua model sebagai berikut:

**Random Forest Regressor**:
* Train MSE: 256.29
* Test MSE: 1864.04
* Train MAE: 7.06
* Test MAE: 18.94
* Train R-squared: 0.978
* Test R-squared: 0.835

**XGBoost Regressor**:
* Train MSE: 1129.57
* Test MSE: 1816.07
* Train MAE: 18.84
* Test MAE: 22.65
* Train R-squared: 0.901
* Test R-squared: 0.839


### **Interpretasi Hasil:**

1. Performa Error (MSE dan MAE):
   * Random Forest menunjukkan performa sangat baik pada data latih (train), dengan error yang sangat rendah. Namun, terjadi lonjakan error cukup besar pada data uji (test), yang mengindikasikan adanya gejala overfitting.
   * XGBoost memiliki nilai MSE dan MAE yang lebih stabil antara train dan test, meskipun error pada train lebih tinggi dari Random Forest. Ini menandakan bahwa model lebih mampu melakukan generalisasi terhadap data baru.

2. R-squared (Koefisien Determinasi) :
   * Random Forest mampu menjelaskan sekitar 97.8% variansi data latih, namun pada data uji hanya menjelaskan 83.5%, memperkuat indikasi bahwa model terlalu menyesuaikan diri pada data pelatihan.
   * XGBoost meski memiliki R² yang lebih rendah pada data latih (90.1%), namun performa lebih konsisten pada data uji (83.9%), menunjukkan bahwa model ini lebih stabil dan tahan terhadap overfitting, serta lebih mampu dalam memahami konteks data baru.

**Kesimpulan**
Berdasarkan metrik evaluasi yang digunakan, XGBoost dapat dianggap sebagai model lebih baik dan stabil dalam konteks prediksi penjualan harian, karena kemampuannya untuk memberikan prediksi akurat dan lebih konsisten pada data yang belum pernah dilihat sebelumnya.

Metrik yang digunakan juga sesuai dengan problem statement dimana MSE dan MAE mampu secara langsung mengukur seberapa jauh hasil prediksi menyimpang dari nilai penjualan yang sebenarnya. Dalam konteks regresi numerik seperti ini, MSE dan MAE memberikan informasi kuantitatif tentang rata-rata kesalahan prediksi, sementara R-squared memberikan gambaran seberapa besar variasi penjualan harian yang berhasil dijelaskan oleh model berdasarkan fitur-fitur input yang tersedia dalam dataset.

**Rubrik Tambahan**

### Metrik Evaluasi yang Digunakan

Dalam proyek prediksi penjualan harian toko ritel menggunakan dataset *Store Sales - Time Series Forecasting* dari Kaggle, saya menggunakan tiga metrik evaluasi utama, yaitu **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, dan **R-squared (R²)**. Ketiga metrik ini digunakan untuk mengevaluasi performa dua model regresi yang dibangun: Random Forest dan XGBoost.

1. Mean Squared Error (MSE)
   * Cara Kerja:
     MSE mengukur rata-rata dari kuadrat selisih antara nilai penjualan aktual (`sales`) dan nilai prediksi model. Langkah-langkah perhitungan MSE adalah sebagai berikut:
     1. Hitung selisih (error) antara nilai aktual dan nilai prediksi untuk setiap baris data.
     2. Kuadratkan setiap error tersebut. Pengkuadratan ini memiliki dua tujuan utama:
        * Untuk menghilangkan efek negatif, sehingga error positif dan negatif tidak saling meniadakan.
        * Untuk memberikan bobot lebih besar kepada error yang lebih besar, sehingga MSE sangat sensitif terhadap prediksi yang jauh meleset.
     3. Hitung rata-rata dari seluruh nilai kuadrat error tersebut.
   * Penjelasan:
     MSE memberikan gambaran seberapa besar rata-rata kesalahan kuadrat yang dilakukan model saat memprediksi penjualan. Nilai MSE yang rendah menunjukkan bahwa model menghasilkan prediksi yang lebih akurat secara keseluruhan. Namun, karena satuan MSE adalah kuadrat dari satuan target , interpretasi angka MSE seringkali kurang intuitif dibanding MAE. Dalam konteks proyek ini, Random Forest menghasilkan MSE sebesar 1864.04 pada data uji, sedangkan XGBoost menghasilkan MSE sebesar 1816.07, yang berarti XGBoost sedikit lebih stabil.

2. Mean Absolute Error (MAE)

   * Cara Kerja:
     MAE menghitung rata-rata dari nilai absolut selisih antara nilai aktual dan nilai prediksi. Prosesnya:
     1. Hitung error antara setiap nilai penjualan aktual dan prediksinya.
     2. Ambil nilai absolut dari error-error tersebut untuk memastikan bahwa semua error bersifat positif.
     3. Hitung rata-rata dari nilai-nilai absolut tersebut.
   * Penjelasan:
     MAE lebih mudah diinterpretasikan dibandingkan MSE karena satuannya sama dengan variabel target, yaitu penjualan (misalnya dalam satuan unit atau dolar). Sebagai contoh, MAE sebesar 20 berarti bahwa, secara rata-rata, prediksi model meleset sebanyak 20 unit penjualan dari nilai sebenarnya. Dalam proyek ini, Random Forest memiliki MAE sebesar 18.94 pada data uji, sementara XGBoost menghasilkan MAE sebesar 22.65. Ini menunjukkan bahwa Random Forest sedikit lebih akurat dalam hal kesalahan rata-rata absolut, namun tetap perlu memperhatikan kestabilan model secara keseluruhan.

3. R-squared (R²)
   * Cara Kerja:
     R² digunakan untuk mengukur sejauh mana model mampu menjelaskan variabilitas dalam data target dibandingkan dengan baseline sederhana (yang hanya memprediksi rata-rata penjualan). Proses perhitungannya:
     1. Hitung total variasi nilai penjualan dari rata-ratanya (SST – Total Sum of Squares).
     2. Hitung jumlah error yang tidak dijelaskan oleh model (SSE – Sum of Squared Errors).
     3. Hitung proporsi variasi yang berhasil dijelaskan model, yaitu 1 dikurangi rasio SSE terhadap SST.
   * Penjelasan:
     R² memberikan wawasan tentang seberapa besar kontribusi fitur-fitur input (seperti promosi, transaksi, harga minyak, dan hari libur) dalam menjelaskan fluktuasi penjualan harian.
     * R² mendekati 1 itu berarti model menjelaskan hampir seluruh variabilitas penjualan.
     * R² mendekati 0 itu berarti model tidak jauh lebih baik dari sekadar menebak rata-rata.
     * R² bisa bernilai **negatif** jika model sangat buruk.

     Dalam proyek ini, Random Forest mencapai R² sebesar 0.8347 pada data uji, sedangkan XGBoost sedikit lebih baik dengan R² sebesar 0.8390, yang menunjukkan bahwa meskipun performa keduanya mendekati, XGBoost memiliki kemampuan sedikit lebih baik dalam menjelaskan variasi penjualan harian secara konsisten.

Secara keseluruhan, dari ketiga metrik ini saling melengkapi dan memberikan evaluasi yang menyeluruh:
* MSE untuk menyoroti prediksi yang sangat meleset,
* MAE untuk memberikan ukuran kesalahan rata-rata yang mudah dipahami, dan
* R² untuk menunjukkan seberapa baik model memanfaatkan informasi dalam fitur untuk menjelaskan variasi data.
* 
  Dengan menggunakan metrik-metrik ini, kita dapat melakukan perbandingan objektif antara dua model dan menyimpulkan bahwa XGBoost lebih unggul dalam stabilitas dan generalisasi terhadap data yang belum pernah dilihat sebelumnya.


**Rubrik Tambahan Prediksi Penjualan**

**Prediksi Penjualan Tertinggi dan Terendah dengan Model XGBoost**

Setelah dilakukan pelatihan dan evaluasi model, XGBoost dipilih sebagai model utama karena menunjukkan performa paling konsisten dan stabil antara data latih dan uji. Model ini digunakan untuk melakukan prediksi penjualan harian, lalu hasilnya dianalisis untuk mengidentifikasi toko-toko dengan penjualan tertinggi dan terendah berdasarkan fitur yang telah diolah dan direkonstruksi kembali (melalui inverse one-hot encoding dan inverse MinMaxScaler).

**Prediksi Penjualan Tertinggi (Top 5)**

Hasil prediksi menunjukkan bahwa 5 toko dengan penjualan tertinggi berasal dari:

* Store number: 1, 11, 44, 10, dan 18
* Kategori produk dominan: BREAD/BAKERY, DAIRY, DELI, dan BEVERAGES
* Lokasi: Mayoritas berlokasi di kota Quito, provinsi Pichincha
* Tipe toko: Beragam (A, B, C, dan D), namun semua menunjukkan performa tinggi.

Ciri khas lain yang ditemukan:

* Nilai onpromotion relatif tinggi, salah satunya mencapai 0.090090, menandakan adanya aktivitas promosi aktif.
* Tidak terjadi pada hari libur nasional (real\_holiday = 0), mengindikasikan bahwa faktor utama pendorong penjualan tinggi adalah lokasi strategis dan promosi, bukan momen hari libur.

**Prediksi Penjualan Terendah (Bottom 5)**

Untuk prediksi penjualan terendah, model menunjukkan:
* Store number: 26, 52, 22, 10, dan 18
* Kategori produk: SEAFOOD, HOME CARE, PREPARED FOODS, BABY CARE, dan BEAUTY
* Lokasi toko: Berada di kota Guayaquil, Manta, dan sebagian di Quito
* State: Guayas, Manabí, dan Pichincha
* Tipe toko: Beragam (D, E, A, B, dan C), tapi tidak terkait langsung dengan rendahnya penjualan.

Ciri lainnya:

* Sebagian besar prediksi rendah ini terjadi pada hari libur nasional (real\_holiday = 1)
* Tidak satu pun memiliki nilai onpromotion, menunjukkan tidak adanya upaya promosi saat itu.

**Kesimpulan Strategis**

* Promosi dan lokasi terbukti sebagai dua faktor dominan dalam mendorong penjualan tinggi.
* Kategori produk juga sangat memengaruhi performa penjualan. Produk seperti roti, susu, dan minuman lebih sering dibeli dibanding seafood atau perawatan bayi.
* Untuk toko dengan performa rendah, disarankan melakukan aktivasi promosi dan evaluasi kategori produk yang ditawarkan pada hari-hari tertentu.


