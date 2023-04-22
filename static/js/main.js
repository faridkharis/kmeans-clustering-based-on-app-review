// ======== upload button ========

const uploadButton = document.getElementById('upload-button');
const customUploadButton = document.getElementById('custom-upload-button');
// const uploadtext = document.getElementById('upload-text');

customUploadButton.addEventListener('click', () => {
    uploadButton.click();
})

uploadButton.addEventListener('change', () => {
    if (uploadButton.value) {
        const path = uploadButton.value;
        const filename = path.replace(/^.*\\/, "");
        customUploadButton.innerHTML = filename;
    } else {
        customUploadButton.innerHTML = "Belum ada file yang dipilih";
    }
})

// $('#proxies').DataTable();