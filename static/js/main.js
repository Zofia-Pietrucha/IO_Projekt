document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-upload");
  const submitButton = document.getElementById("submit-button");
  const imagePreview = document.getElementById("image-preview");
  const previewContainer = document.querySelector(".preview-container");
  const resultContainer = document.getElementById("result-container");
  const resultImage = document.getElementById("result-image");
  const predictionResult = document.getElementById("prediction-result");
  const predictionMessage = document.getElementById("prediction-message");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceText = document.getElementById("confidence-text");
  const explanation = document.getElementById("explanation");

  fileInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      resultContainer.classList.add("d-none");

      // Pokaż podgląd wybranego zdjęcia
      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        previewContainer.classList.remove("d-none");
      };
      reader.readAsDataURL(file);
    } else {
      previewContainer.classList.add("d-none");
    }
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      alert("Proszę wybrać plik.");
      return;
    }

    const originalButtonText = submitButton.innerHTML;
    submitButton.innerHTML =
      '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analizowanie...';
    submitButton.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Odpowiedź serwera: " + response.status);
        }
        return response.json();
      })
      .then((data) => {
        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;

        displayResults(data);
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Wystąpił błąd podczas analizy. Spróbuj ponownie.");

        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;
      });
  });

  function displayResults(data) {
    if (data.error) {
      alert("Błąd: " + data.error);
      return;
    }

    resultImage.src = URL.createObjectURL(fileInput.files[0]);

    if (data.prediction === "Melanoma") {
      predictionResult.textContent = "Podejrzenie czerniaka";
      predictionResult.className = "mb-2 text-danger";
      confidenceBar.className = "progress-bar high-risk";
      explanation.textContent =
        "Na przesłanym zdjęciu wykryto cechy, które mogą wskazywać na czerniaka (melanoma). Czerniak jest poważnym nowotworem skóry, który wymaga szybkiej diagnozy i leczenia. Im wcześniej zostanie wykryty, tym lepsze są rokowania.";
    } else {
      predictionResult.textContent = "Prawdopodobnie znamię łagodne";
      predictionResult.className = "mb-2 text-success";
      confidenceBar.className = "progress-bar low-risk";
      explanation.textContent =
        "Na przesłanym zdjęciu nie wykryto wyraźnych cech czerniaka. Większość znamion skórnych to zmiany łagodne, jednak warto regularnie je obserwować i reagować na jakiekolwiek zmiany.";
    }

    predictionMessage.textContent = data.message;
    confidenceBar.style.width = data.confidence + "%";
    confidenceText.textContent = `Pewność klasyfikacji: ${data.confidence.toFixed(
      1
    )}%`;

    resultContainer.classList.remove("d-none");

    resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
