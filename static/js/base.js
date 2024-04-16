
/* Datatables */
$(document).ready(function() {
    $('#ds-table').find('table').each(function() {
        $(this).DataTable({
            "language": {
                "url": "//cdn.datatables.net/plug-ins/1.11.3/i18n/Portuguese.json"
            }
        });
    });
});

/* Create Model Form */
document.getElementById('algo-type').addEventListener('change', function() {
    // Hide all form sections
    document.querySelectorAll('.form-section').forEach(function(section) {
        section.style.display = 'none';
    });

    // Hide the submit button
    document.getElementById('submit-bt-div').style.display = 'none';

    // Show the selected form section
    var selectedValue = this.value;
    if (selectedValue) {
        document.getElementById('form-' + selectedValue).style.display = 'block';

        // Show the submit button
        document.getElementById('submit-bt-div').style.display = 'block';
    }
});




/* Sidebar JS */
/* const body = document.querySelector('body'),
    sidebar = document.querySelector('.sidebar'),
    toggle = document.querySelector('.toggle');

sidebar.addEventListener('mouseover', () => {
        sidebar.classList.remove("close");
    });
    
sidebar.addEventListener('mouseout', () => {
        sidebar.classList.add("close");
    }); */

/* document.addEventListener('DOMContentLoaded', function() {
    var rows = document.querySelectorAll('tr.clickable-tr');

    rows.forEach(function(row) {
        row.addEventListener('click', function() {
            window.location = this.getAttribute('data-href');
        });
    });
}); */