# Diferansiyel Robot Navigasyonu — Çok Yöntemli Paralel Simülasyon

Bu proje, bilinmeyen bir çevrede hedefe ulaşmaya çalışan diferansiyel sürüşlü bir mobil robotun hedefeulaşma problemini ele almaktadır.
Simülasyon, robotun anlık sensör verileriyle haritayı keşfetmesini, engellerden kaçınmasını ve 5 farklı yol planlama algoritmasını 
kullanarak karşılaştırmasını sağlar.

---

## 1. Giriş ve Senaryo Tanımı

Bu simülasyonda robot, 20-20 metrelik bir 2D dünya üzerinde önceden bilmediği bir ortamda başlangıç noktasından (1.5, 1.5 konum)
hedef noktasına (18.5, 18.5 konum) ulaşmaya çalışır. 

==Çalışma Senaryosu:==
- Robot haritanın tamamını bilmez; yalnızca hedefin global koordinatlarını bilir.
- Üzerinde bulunan 72 ışınlı (ray) ve 6 metre menzilli 'LiDAR'sensörü ile çevresini anlık olarak tarar.(Kod üstünden değerler değişebilir.)
- Karşılaştığı her yeni engeli kendi iç haritasına kaydeder ve mevcut yolun kapanması durumunda rotasını anlıkolarak yeniden planlar.
- Çarpışmaları önlemek için robotun fiziksel yarıçapı (r = 0.15m) dikkate alınarak harita üzerinde uzay üzerinde 
şişirmesi yapılır.(Çarpma önlemek için)
- Kontrolcü, dinamik engellerden veya dar koridorlardan geçerken otonom yavaşlama ve hızlanma uygulayarak güvenli sürüşsağlar.
Robotun sıkışması durumunda 90 derecelik kurtarma manevrası otomatik devreye girer.

---

## 2. Kullanılan Yöntemler

==Yol Planlama Algoritmaları==
Yazılım, aşağıdaki yöntemleri aynı anda çalıştırarak performanslarını kıyaslar:
1. D*Lite: Hedefe doğru geriye dönük arama yapan, değişen ortamlarda tüm haritayı yeniden hesaplamak yerine sadece değişen
 kısımları güncelleyen eklemeli ve hızlı planlayıcı.
2. A*(A-Star): Yeni bir engel tespit edildiğinde mevcut konumdan hedefe kadar en kısa yolu sezgisel bir yaklaşımla sıfırdan hesaplayan 
geleneksel yöntem.
3. Dijkstra: Sezgisel bir fonksiyon kullanmayan, maliyet odaklı çalışan kapsamlı temel arama algoritması.
4. RRT: Izgara üzerinde rastgele örnekleme yaparak ağaç yapısında hızla yayılan, özellikledar koridorlarda ve sürekli alanlarda etkili 
çalışan olasılıksal planlayıcı.
5. PRM:** Bilinen boş alanlara rastgele düğümler yerleştirip bunları birleştirerek bir yol haritası çıkartan ve ardından bunun
üzerinde A* ile en kısa yolu bulan algoritma.

==Konum Kestirimi (Lokalizasyon) ve Filtreleme==
-Genişletilmiş Kalman Filtresi (EKF):** Robotun anlık pozisyonu (x, y, q); enkoder kaymaları, enkoder gürültüsü ve IMU 
sürüklenme hataları barındırır. EKF, bu hatalı "Ölü Hesap" verisini LiDAR bazlı konum doğrulama ile birleştirerek güvenilir 
bir konum tahmini üretir.

---

## 3. Sonuçlar ve Grafikler

Simülasyon, "Dağınık", "Koridor" ve "Labirent" olmak üzere 3 farklı harita seçeneği sunar. Çalışma esnasında 'matplotlib' tabanlı 
interaktif bir GUI üzerinden her algoritmanın robota yaptırdığı hareketler eşzamanlı izlenir.

==Elde Edilen Çıktılar ve Görselleştirmeler:==
1. Canlı Simülasyon Ekranı:5 algoritmanın bulunduğu bir ızgarada; keşfedilen harita, EKF kestirim yolu, planlanan anlık rota ve 
   LiDAR ışınları canlı olarak render edilir.
2. Detay Analiz Paneli: Her algoritmaya özel açılabilen bu ekranda;
   -LiDAR'ın ham ve medyan filtrelenmiş tarama grafikleri,
   -Gerçek yol, EKF tahmini ve Ölü Hesap (DR) yolunun 2D iz düşümü,
   -Zaman bazlı x, y ve q sapma grafikleri,
   -EKF ve DR hatalarının RMSE ve MAE değerleri sunulur.
3. -Genel Karşılaştırma Grafikleri:Tüm araçlar hedefe ulaştığında, algoritmaların Gerçek Yol Uzunluğu,Ulaşma Süresi ve Yeniden 
   Planlama Sayısı değerlerini karşılaştıran sütun grafikleri üretilir.

---

## 4. Hata Analizi ve Kısa Tartışma

==Simülasyon sonuçlarına göre aşağıdaki analizler yapılmıştır==

-Algoritma Performansları:Klasik yöntemler (A*, Dijkstra) dinamik bir engelle karşılaştıklarında tüm haritayı baştan taradıkları
 için CPU yükünü artırır. D*Lite ise lokal güncellemeler yaptığı için en verimli olandır.
 Olasılıksal yöntemler (RRT, PRM) hedefe ulaşsalar da rastgele örneklemeden dolayı her simülasyonda farklı yol uzunlukları üretir ve
 nadiren optimal yolu bulurlar.
-Lokalizasyon Hataları:Sadece enkoder ve IMU verisine güvenildiğinde, IMU ve tekerlek kayma katsayısı nedeniyle kümülatif bir pozisyon 
hatası oluşur.Zamanla robotun haritadaki gerçek konumu ile inandığı konum arasında metrelerce fark çıkar.EKF ise yakın engellerden alınan 
LiDAR referans ölçümleriyle pozisyonu düzelterek hatayı kabul edilebilir bir sınırın altında tutmayı başarır.
-Kontrol ve Sıkışma Durumları:Dar koridorlara girildiğinde kontrolcünün her iki taraftan aldığı "itme"kuvvetleri robotu osilasyona 
sokabilir.Simülasyonda uygulanan "2.5 saniyede ilerleyememe durumunda yolu sıfırla ve 90° dön" mekanizması 
bu tür yerel minimum durumlarından robotun kurtulmasını sağlamıştır.

---

## 5. Yapay Zeka Kullanım Beyanı

==Yapay Zeka Kullanım Beyanı==
Bu projenin kaynak kodlarının yapılması,algoritmaların (D*, A*, Dijkstra, PRM, RRT) Python ile yapılmıştır.
