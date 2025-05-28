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


