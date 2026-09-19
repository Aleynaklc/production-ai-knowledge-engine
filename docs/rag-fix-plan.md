# RAG düzeltme planı

Mevcut 0.5B üretici model korunacak. Değişiklikler önce izole test korpusunda ve gerçek belgelerin geçici kopyasında değerlendirilecek; ölçüm başarısızlıkları saklanacak.

- [x] 1. Belge seçimi: otomatik tek kelimelik zorunlu filtreyi kaldır; açık belge adını ve içerik eşleşmesini ayır. Aday havuzunu büyüt, sözcük varyasyonlarını destekle, soru başlıklarını koru.
- [x] 2. Çıkarım: PDF okuma sırası, DOCX üst/altbilgi ve bölüm bütünlüğü; taranmış sayfalar için sınırlı yerel OCR ve açık hata davranışı.
- [x] 3. Chunking: gerçek token sınırı, uzun metinde örtüşme, kod/tablo bloklarının korunması, komşu parçaların kaynak kimliği korunarak birleştirilmesi.
- [x] 4. Context: gerçek okuma biçimiyle token bütçesi, top_k ile kaynak sayısının birlikte çalışması, üretici giriş sınırı, kesilme tanıları.
- [x] 5. Yanıt: kapsam ve belirsizlik, kaynak-soru ilişkisinin kontrolü, doğrulanmış alıntı yolu, gereksiz yanlış retlerin ayrıştırılması; kaynak dışı cevaplara karşı negatif testler.
- [x] 6. Ölçüm: belge kimliğine ek olarak cevap taşıyan metin kapsamı; çok belgeli/Türkçe/kod/olmayan bilgi testleri. Önceki 40 senaryoyla karşılaştırma.
- [ ] 7. Geçiş: mevcut belgelere yeniden indeksleme, cache geçersizleştirme, tam testler, yerel servis ve kullanıcı akışının kontrolü.

Tam doğruluk varsayılmayacak. Sağlanamayan model/semantik doğrulama hedefleri, yazılım düzeltmelerinden ayrı olarak sonuç raporunda açık kalacak.

## Doğrulanan sonuçlar

- Nihai tam Python test paketi: 247 test geçti (inline-citation regresyonu dahil).
- Ruff ve mypy temiz; frontend lint, TypeScript ve üretim derlemesi başarılı.
- Aynı 0.5B modelle son 40 senaryo: 28/32 olumlu başarı (%87,5), cevabı olmayan 8 soruda 0 yanlış yanıt. Etiketli cevap parçalarının gerçek context içindeki kapsamı %100.
- Önceki v2: 22/32 olumlu başarı ve 3/8 yanlış yanıt. Kod/prosedür alıntısı ile doğrulanmış hesaplama yeni ayrı cevap yollarıdır; modelin semantik yeteneğinin aynı oranda arttığı iddia edilmez.
- Mevcut PDF ve Markdown belge sürüm 2'ye taşındı; sırasıyla 7 ve 556 chunk/vektör hazır. Önceki kaynak sürümleri korunuyor.
- Tarayıcı giriş ekranı açılıyor, API erişilebilir. Oturum bulunmadığından giriş sonrası görsel akış kontrolü bekliyor; belge seçimi ve yetkilendirme API testlerinde doğrulandı.

## Açık kalan işler

Son değerlendirmede kod açıklaması, runbook özeti, tarihli politikaların çelişkisini açıklama ve Türkçe kısa sayısal cevabı doğrulama olmak üzere dört olumlu senaryo başarısız. Bu maddeler tamamlanmış sayılmıyor. Ayrıntılar ve ölçümün sınırları [kalite raporunda](workspace-quality.md); makine çıktısı [v6 raporunda](../evaluation/reports/workspace_scenarios_v6.json).

Gerçek belge kopyalarıyla son kontrol: NeetCode örnek isteği kaynak kod alıntısıyla cevaplanıyor ve belgedeki olmayan garanti bilgisi reddediliyor. Kimlik ve iki kod açıklama/çıktı sorusu hâlâ başarısız; üniversite listesi bir dil okulunu da içeriyor. Bunlar 40 sentetik senaryoya ek açık doğruluk sorunlarıdır.

## Genelleme çalışması — 2026-09-16

Yeni 24 soruluk kabul kümesi, veri özetiyle sabitlendi ve geliştirmedeki 40 sorudan ayrıldı. Bölüm sınırı denemesi yeni sorularda 14/18 sonucunu 13/18'e düşürdüğü için geri alındı. Korunan sürüm tekrar 14/18 ve geliştirmede 28/32 ölçüldü; bu tur için doğruluk artışı iddia edilmiyor. Yeni ölçüm, regresyon denetimi, API hata mesajı ayrımı ve belge-adı doğrulama düzeltmesi projede kaldı. Çok dilli anlamsal doğrulama ve belirsizlik yönetimi hâlâ açık. [Deneyler ve sonraki aşamalar](generalization-roadmap.md).
