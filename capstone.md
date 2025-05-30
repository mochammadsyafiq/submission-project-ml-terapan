# Technologies

## Machine Learning

* keras
* matplotlib
* Pillow
* protobuf
* scikit\_learn
* seaborn
* tensorflow
* tensorflowjs

**Machine Learning untuk Klasifikasi Sampah Kompos**

Proyek ini kami kembangkan menggunakan **Google Colab** karena keterbatasan spesifikasi perangkat PC/Laptop. Tujuannya adalah membangun model klasifikasi gambar untuk membedakan jenis sampah berdasarkan kelayakannya untuk dijadikan kompos. Model ini mengelompokkan gambar sampah ke dalam tiga kelas utama:

1. **Sampah Organik Basah (Layak Kompos)**
2. **Sampah Organik Kering (Layak Kompos)**
3. **Sampah Tidak Layak Kompos**

Dataset yang kami gunakan terdiri dari **12.464 gambar sampah**, yang terbagi menjadi:

* 4.451 gambar sampah organik basah
* 4.009 gambar sampah organik kering
* 4.004 gambar sampah tidak layak kompos

Semua gambar disimpan di **Google Drive** dan selanjutnya diproses untuk pemrosesan model. Pertama yang kami lakukan yaitu pembagian data yang dimana dilakukan dengan menggunakan metode **stratified splitting**, yaitu membagi data sambil menjaga proporsi gambar dari setiap kelas tetap seimbang. Data dibagi menjadi tiga bagian:

* **80% untuk pelatihan (training)**
* **10% untuk validasi (validation)**
* **10% untuk pengujian (testing)**

Setelah dilakukan splitting data atau pembagian data, semua gambar dikonversi ke **format RGB** dan di-*resize* menjadi ukuran **512x512 piksel** agar seragam dan sesuai untuk pemrosesan model.

Dataset gambar yang sudah melalui proses splitting data kemudian diproses menggunakan metode **preprocessing MobileNetV2**, yaitu mengubah nilai piksel dari rentang \[0,255] ke \[-1,1] agar sesuai dengan standar model pretrained.

Model klasifikasi dibangun menggunakan teknik **transfer learning** dengan memanfaatkan **arsitektur MobileNetV2** sebagai feature extractor. Beberapa layer tambahan ditambahkan untuk mengoptimalkan klasifikasi:

* **Global Average Pooling** untuk merangkum fitur
* **Dropout** untuk mencegah overfitting
* **Dense layer dengan softmax** untuk menghasilkan hasil prediksi akhir dalam bentuk probabilitas

Model dilatih menggunakan **Adam Optimizer** dengan *learning rate* kecil agar proses pelatihan lebih stabil. Fungsi loss yang digunakan adalah **Categorical Crossentropy Loss**, karena klasifikasi bersifat multi-kelas. Beberapa teknik tambahan digunakan untuk meningkatkan performa dan efisiensi pelatihan yang dimana ini juga saran dari Advisor untuk menghindari *overfitting*, seperti:

> Model menggunakan callback **stop\_in\_accuracy**, **ReduceLROnPlateau**, dan **ModelCheckpoint**. yang dimana fungsinya untuk Callback `stop_in_accuracy` secara otomatis menghentikan pelatihan jika akurasi dan val\_akurasi sudah mencapai 95%, sementara dua callback lainnya berguna untuk mengoptimalkan efisiensi pelatihan dan menyimpan model terbaik.

Setelah pelatihan, model terbaik disimpan dalam dua format: **`.h5`** dan **SavedModel**, agar fleksibel digunakan kembali atau dikonversi ke platform lain.

Model diuji menggunakan data pengujian dan berhasil mencapai **akurasi lebih dari 95%**. Evaluasi performa dilakukan dengan:

* Menampilkan grafik akurasi selama pelatihan
* Melihat hasil klasifikasi dalam bentuk **confusion matrix**
* Menggunakan **classification report** untuk melihat metrik precision, recall, dan F1-score dari masing-masing kelas

Untuk bisa digunakan di web, model kemudian dikonversi ke format **TensorFlow\.json (TFJS)** sehingga bisa dijalankan langsung di browser tanpa perlu server backend. Label kelas juga disimpan dalam file `labels.json` untuk memudahkan integrasi dengan tampilan antarmuka web.



