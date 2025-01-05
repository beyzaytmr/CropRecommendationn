document.getElementById('crop-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    // Form verilerini al
    const formData = {
        Nitrogen: document.getElementById('Nitrogen').value,
        Phosphorus: document.getElementById('Phosphorus').value,
        Potassium: document.getElementById('Potassium').value,
        Temperature: document.getElementById('Temperature').value,
        Humidity: document.getElementById('Humidity').value,
        pH_Value: document.getElementById('pH_Value').value,
        Rainfall: document.getElementById('Rainfall').value,
    };

    try {
        // Flask API'ye POST isteği gönder
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        });

        const result = await response.json();

        // Tahmin sonucunu göster
        if (response.ok) {
            document.getElementById('result').innerHTML = `Önerilen Ürün: ${result.predicted_crop}`;
        } else {
            document.getElementById('result').innerHTML = `Hata: ${result.error}`;
        }
    } catch (error) {
        document.getElementById('result').innerHTML = 'Bir hata oluştu.';
        console.error(error);
    }
});
document.body.style.backgroundImage = "url('.soil.jpg')";
document.body.style.backgroundSize = "cover";
document.body.style.backgroundPosition = "center";