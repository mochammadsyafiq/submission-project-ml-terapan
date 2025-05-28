# Laporan Proyek Machine Learning - Mochammad Syafiq Ilallah

## Project Overview

Dalam era digital yang dipenuhi informasi, pengguna sering kali menghadapi kesulitan dalam memilih konten yang sesuai dengan preferensi mereka, terutama dalam platform layanan hiburan seperti Netflix, Disney+, dan IMDb. Untuk mengatasi masalah ini, sistem rekomendasi menjadi komponen penting dalam menyajikan pengalaman pengguna yang lebih personal. Dua pendekatan populer yang digunakan dalam sistem rekomendasi adalah **Content-Based Filtering** dan **Collaborative Filtering**.

**Content-Based Filtering** bekerja dengan menganalisis atribut atau konten dari item (seperti genre atau sinopsis film) untuk merekomendasikan item serupa yang pernah disukai pengguna. Sementara itu, **Collaborative Filtering** menganalisis pola perilaku pengguna lain yang memiliki preferensi serupa untuk memberikan rekomendasi. Keduanya memiliki kelebihan dan kekurangan masing-masing, dan penelitian mengenai kombinasi atau evaluasi dari kedua pendekatan ini menjadi hal penting dalam meningkatkan akurasi dan relevansi sistem rekomendasi.

### **Kriteria Tambahan**
1. **Urgensi dan Pentingnya Proyek**

Proyek ini penting untuk diselesaikan karena sistem rekomendasi yang baik dapat secara signifikan meningkatkan kepuasan pengguna dan efisiensi pencarian informasi. Dengan meningkatnya volume data dan kompleksitas preferensi pengguna, pendekatan tunggal sering kali tidak cukup. Oleh karena itu, mengeksplorasi dan mengevaluasi dua model yang berbeda—**Content-Based Filtering** dan **Neural Collaborative Filtering**—dapat memberikan wawasan mengenai performa keduanya dalam kasus nyata, seperti dataset **MovieLens**.

Selain itu, penelitian terbaru menunjukkan bahwa sistem rekomendasi dapat mengandung **bias** atau menghasilkan rekomendasi yang **tidak adil** jika hanya mengandalkan salah satu pendekatan (González et al., 2022). Oleh karena itu, penting untuk memahami bagaimana setiap model bekerja, kekuatannya, serta keterbatasannya dalam konteks data nyata dan beragam.

2. **Hasil Riset dan Referensi Terkait**

Berbagai studi telah mengkaji efektivitas dari metode rekomendasi. Harper dan Konstan (2015) menjelaskan bahwa MovieLens adalah dataset benchmark yang umum digunakan untuk mengembangkan dan menguji sistem rekomendasi karena kualitas dan kelengkapan datanya. Sementara itu, Mu dan Wu (2023) menunjukkan bahwa pendekatan multimodal berbasis deep learning dapat mengungguli metode tradisional jika diterapkan dengan tepat.

Penelitian oleh Abbas dan Khan (2020) juga mendemonstrasikan efektivitas deep learning dalam meningkatkan akurasi prediksi rating pengguna. Hal ini sejalan dengan Chen et al. (2019) yang mengembangkan model **Joint Neural Collaborative Filtering**, sebuah pendekatan yang menyatukan keunggulan dua metode dalam satu jaringan saraf.

3. **Referensi**

* Abbas, H., & Khan, A. (2020). An efficient deep learning approach for collaborative filtering recommender systems. *Procedia Computer Science, 167*, 2621–2628. [https://doi.org/10.1016/j.procs.2020.03.239](https://doi.org/10.1016/j.procs.2020.03.239)
* Chen, W., Cai, F., Chen, H., & de Rijke, M. (2019). Joint neural collaborative filtering for recommender systems. *arXiv preprint* arXiv:1907.03459. [https://doi.org/10.48550/arXiv.1907.03459](https://doi.org/10.48550/arXiv.1907.03459)
* González, Á., Ortega, F., Pérez-López, D., & Alonso, S. (2022). Bias and unfairness of collaborative filtering-based recommender systems in MovieLens dataset. *IEEE Access, 10*, 80094–80108. [https://doi.org/10.1109/ACCESS.2022.3186719](https://doi.org/10.1109/ACCESS.2022.3186719)
* Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and context. *ACM Transactions on Interactive Intelligent Systems (TiiS), 5*(4), 1–19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)
* Mu, Y., & Wu, Y. (2023). Multimodal movie recommendation system using deep learning. *Mathematics, 11*(4), 895. [https://doi.org/10.3390/math11040895](https://doi.org/10.3390/math11040895)

## **Business Understanding**

### **Problem Statements**

1. **Bagaimana membuat sistem rekomendasi film yang dipersonalisasi berdasarkan metadata film (seperti genre) dan preferensi pengguna menggunakan pendekatan content-based filtering?**
2. **Bagaimana merekomendasikan film lain yang mungkin disukai oleh pengguna, meskipun belum pernah ditonton sebelumnya, dengan pendekatan collaborative filtering berdasarkan data rating?**

### **Goals**

1. Menghasilkan sejumlah rekomendasi film yang **dipersonalisasi berdasarkan genre** yang disukai pengguna menggunakan **content-based filtering**.
2. Menghasilkan sejumlah rekomendasi film yang **belum ditonton, tetapi berpotensi disukai** oleh pengguna, berdasarkan **pola perilaku pengguna lain** menggunakan **collaborative filtering**.

### **Kriteria Tambahan**

**Solution Approach**

Untuk mencapai tujuan di atas, proyek ini menggunakan dua pendekatan sistem rekomendasi yang berbeda:

1. **Content-Based Filtering**

* **Pendekatan**: Menggunakan metadata film, khususnya genre, untuk membangun representasi numerik menggunakan **TF-IDF Vectorizer**.
* **Proses**:

  * Setiap film diubah menjadi vektor berdasarkan genre-nya.
  * Menggunakan **Cosine Similarity** untuk mengukur kemiripan antar film.
  * Rekomendasi diberikan berdasarkan kemiripan genre dengan film yang sebelumnya disukai oleh pengguna.
* **Kelebihan**:

  * Tidak bergantung pada aktivitas pengguna lain.
  * Dapat merekomendasikan film baru dengan genre serupa.

2. **Collaborative Filtering (Neural Embedding)**

* **Pendekatan**: Menggunakan data interaksi pengguna (user–movie–rating) dan membangun model neural network dengan embedding.
* **Proses**:

  * Encoding numerik pada `userId` dan `movieId`.
  * Membangun model neural network (RecommenderNet) dengan **embedding layer** untuk menangkap pola latennya.
  * Model dipelajari untuk memprediksi rating yang akan diberikan user terhadap film tertentu.
  * Film dengan prediksi tertinggi yang belum pernah ditonton direkomendasikan ke pengguna.
* **Kelebihan**:

  * Lebih akurat dalam memahami selera personal.
  * Dapat merekomendasikan film yang sangat berbeda kontennya tapi relevan secara preferensi pengguna.


## **Data Understanding**

### **Sumber Data**

Dataset yang digunakan dalam proyek ini adalah **MovieLens Latest Small Dataset** yang tersedia secara publik di tautan berikut:
🔗 [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

Dataset ini banyak digunakan sebagai benchmark dalam pengembangan sistem rekomendasi karena kualitas anotasi dan struktur datanya (Harper & Konstan, 2015).

### **Informasi Umum Dataset**

Dataset ini berisi **100.836 rating** yang diberikan oleh **610 pengguna** terhadap **9.742 film**. Setiap pengguna memberikan rating minimal 20 film.

Dataset terdiri dari beberapa file utama:

| Nama File     | Deskripsi                                                                |
| ------------- | ------------------------------------------------------------------------ |
| `movies.csv`  | Informasi tentang film seperti `movieId`, `title`, dan `genres`.         |
| `ratings.csv` | Data interaksi pengguna: `userId`, `movieId`, `rating`, dan `timestamp`. |
| `tags.csv`    | Tag atau anotasi dari pengguna terhadap film tertentu (opsional).        |
| `links.csv`   | Tautan `movieId` ke ID di IMDb dan TMDb.                                 |


### **Kriteria Tambahan**
**Exploratory Data Analysis (Univariate - Non-Visual)**

1. **Struktur Dataset**

Metode: `DataFrame.info()`

* Digunakan untuk melihat jumlah data, tipe data, dan nilai null pada masing-masing kolom.
* Insight:

  * Tidak ada missing value di `movies`, `ratings`, dan `tags`.
  * Terdapat 8 missing values di kolom `tmdbId` pada `links`.

2. **Entitas Unik**

Metode: `len().unique()`

* Jumlah film unik: **9.742**
* Jumlah pengguna yang memberikan rating: **610**
* Jumlah rating: **100.836**
* Jumlah pengguna yang memberi tag: **609**
* Jumlah film yang diberi tag: **1.828**
* Jumlah tag unik: **1.293**
* Jumlah film yang memiliki link eksternal: **9.742**

Insight:

* Tidak semua pengguna aktif menggunakan fitur tagging.
* Relasi ke IMDb/TMDb sangat lengkap.

3. **Duplikasi**

Metode: `duplicated().sum()`

* Semua file (`ratings`, `movies`, `tags`, `links`) tidak memiliki duplikat.
* Insight: Data bersih dan siap untuk proses modeling.

4. **Eksplorasi Variabel**

**a. `movies.csv`**

* Jumlah film: 9.742
* Genre tersedia dalam format string terpisah tanda “|”.
* Insight:

  * Genre cukup beragam dan menjadi basis utama untuk *Content-Based Filtering*.
  * Tidak ada missing value.

**b. `ratings.csv`**

* Rating dari 0.5 – 5.0 dalam interval 0.5.
* Total: 100.836 rating dari 610 pengguna untuk 9.724 film.
* Insight:

  * Skala rating granular dan cocok untuk *Collaborative Filtering*.

**c. `tags.csv`**

* Total tag: 3.683 dari 609 pengguna.
* Insight:

  * Penggunaan tag bersifat opsional dan terbatas, tapi bisa dipakai untuk analisis semantik lanjutan.

**d. `links.csv`**

* Total film: 9.742
* Ada 8 missing value di `tmdbId`.
* Insight:

  * Penting untuk integrasi metadata eksternal.

## **Data Preprocessing**

Tahap ini bertujuan untuk mempersiapkan data agar layak digunakan dalam proses pemodelan sistem rekomendasi. Data preprocessing dilakukan secara sistematis untuk memastikan integritas, kebersihan, dan keterpaduan data.

1. Konversi Kolom Waktu

Langkah pertama melibatkan konversi kolom `timestamp` pada data `ratings` dan `tags` dari format Unix time ke format `datetime` menggunakan fungsi `pd.to_datetime()` dengan parameter `unit='s'`. Konversi ini memudahkan pembacaan waktu oleh manusia dan membuka kemungkinan analisis berbasis waktu di tahap lanjutan.

2. Penggabungan Dataset

Seluruh file utama dalam dataset digabungkan untuk membentuk satu kesatuan data yang komprehensif.

* Dataset `movies` digabungkan dengan `links` menggunakan *left join* berdasarkan `movieId` untuk menambahkan informasi ID eksternal seperti `imdbId` dan `tmdbId`.
* Selanjutnya, hasil penggabungan tersebut digabungkan dengan dataset `ratings` untuk mengintegrasikan informasi interaksi pengguna.
* Akhirnya, dataset `tags` ditambahkan berdasarkan kombinasi `movieId` dan `userId`.

Penggabungan ini bertujuan untuk menciptakan dataset yang mengandung informasi film, interaksi pengguna, dan preferensi tambahan berupa tag.

Berikut adalah **revisi bagian “Penghapusan Kolom Redundan”** dengan pilihan kata yang lebih tepat dan sesuai gaya APA:

3. Penghapusan Kolom yang Tidak Relevan

Setelah proses penggabungan beberapa dataset, kolom `timestamp_x` dan `timestamp_y` yang berasal dari tabel `ratings` dan `tags` dihapus menggunakan fungsi `drop()`. Kedua kolom tersebut merupakan hasil duplikasi dari atribut waktu dan tidak memiliki peran signifikan dalam proses pemodelan. Oleh karena itu, penghapusannya bertujuan untuk menyederhanakan struktur data, mencegah kebingungan dalam analisis, serta menjaga efisiensi pemrosesan (Han, Kamber, & Pei, 2011).

4. Penanganan Missing Values

Penanganan nilai kosong (missing values) dilakukan secara selektif:

* Kolom `tag` diisi dengan string kosong (`''`) karena bersifat opsional.
* Baris yang memiliki nilai kosong pada kolom `rating`, `userId`, atau `tmdbId` dihapus. Hal ini penting agar setiap entri memiliki informasi rating dan identitas pengguna yang lengkap, serta terhubung ke sumber metadata eksternal.

5. Pembersihan Judul Film

Kolom `title` pada dataset film dibersihkan dari informasi tahun rilis menggunakan teknik ekspresi reguler (`re.sub`). Tahun rilis yang biasanya ditulis dalam format "(YYYY)" di akhir judul dihapus untuk meningkatkan konsistensi saat digunakan dalam proses tokenisasi atau pencocokan teks.

6. Penyusunan Data Film Unik

Dataset kemudian diurutkan berdasarkan `movieId` dan difilter untuk memastikan hanya ada satu entri unik per film. Data yang telah dibersihkan disimpan dalam DataFrame baru bernama `movies_new`, yang memuat tiga kolom utama: `movieId`, `movie_title`, dan `genre`. Dataset ini digunakan sebagai referensi utama dalam pengembangan model rekomendasi berbasis konten.

### **Alasan dan Justifikasi**

Tahapan preprocessing ini diperlukan untuk:

* Membersihkan data yang tidak diperlukan dan menghindari data ganda.
* Memastikan keterpaduan antar sumber informasi (film, pengguna, interaksi).
* Menyiapkan data agar kompatibel dengan dua pendekatan model yang digunakan: *Content-Based Filtering* dan *Collaborative Filtering*.
* Mengoptimalkan kualitas input yang akan diberikan ke model machine learning, yang sangat bergantung pada kebersihan dan kelengkapan data input (Han, Kamber, & Pei, 2011).


