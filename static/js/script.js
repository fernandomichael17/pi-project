document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const loader = document.getElementById('loader');

    form.addEventListener('submit', function() {
        loader.style.display = 'block';
    });
});
