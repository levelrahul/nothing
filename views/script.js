const uploadForm = document.getElementById('upload');
const resultDiv = document.getElementById('result');

uploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            const prediction = response.prediction;
            resultDiv.innerHTML = `Prediction Result: ${prediction}`;
        }
    };
    xhr.send(formData);
});