
/* Datatables */
/* $(document).ready(function() {
    $('#ds-table').find('table').each(function() {
        $(this).DataTable({
            "language": {
                "url": "//cdn.datatables.net/plug-ins/1.11.3/i18n/Portuguese.json"
            }
        });
    });
}); */


/* Create Model Form */
/* document.getElementById('algo-type').addEventListener('change', function() {
    // Hide all form sections
    document.querySelectorAll('.form-section').forEach(function(section) {
        section.style.display = 'none';
    });

    // Hide the submit button
    document.getElementById('submit-bt-div').style.display = 'none';

    // Show the selected form section
    const selectedValue = this.value;
    if (selectedValue) {
        document.getElementById('form-' + selectedValue).style.display = 'block';

        // Show the submit button
        document.getElementById('submit-bt-div').style.display = 'block';
    }
}); */


/* show advanced options button and the basic options only when kernel is selected */
document.getElementById('kernel').addEventListener('change', function() {

    const selectedOption = this.value;

    // get toggle button
    const button = document.getElementById('toggle-advanced-svm-options');

    // get the basic options
    const basic_SVM_Options = document.getElementById('basic-svm-options');
    const basic_LSVM_Options = document.getElementById('basic-lsvm-options');

    // get the advanced options div, they should be hidden when changing kernel
    const advanced_SVM_Options = document.getElementById('advanced-svm-options');
    const advanced_LSVM_Options = document.getElementById('advanced-lsvm-options');

    // Show the button if 'Linear', 'Polinomial', or 'Função de base radial' is selected
    if (selectedOption === '1' || selectedOption === '2' || selectedOption === '3') {
        button.style.display = 'block';
    } else {
        button.style.display = 'none';
    }

    // Show the basic options if 'Linear' is selected
    if (selectedOption === '1') {
        basic_LSVM_Options.style.display = 'block';
    } else {
        basic_LSVM_Options.style.display = 'none';
    }

    // Show the basic options if 'Polinomial' or 'Função de base radial' is selected
    if (selectedOption === '2' || selectedOption === '3') {
        basic_SVM_Options.style.display = 'block';
    } else {
        basic_SVM_Options.style.display = 'none';
    }

    // Hide the advanced options when changing kernel
    if (advanced_SVM_Options.style.display === 'block') {
        advanced_SVM_Options.style.display = 'none';
    }
    if (advanced_LSVM_Options.style.display === 'block') {
        advanced_LSVM_Options.style.display = 'none';
    }
});


/* Toggle advanced SVM options */
document.getElementById('toggle-advanced-svm-options').addEventListener('click', function() {
    
    // get the options div
    const advanced_SVM_Options = document.getElementById('advanced-svm-options');
    const advanced_LSVM_Options = document.getElementById('advanced-lsvm-options');

    // get the selected kernel
    const selectedKernel = document.getElementById('kernel').value;

    switch(selectedKernel) {
        case '1':
            // 'Linear' is selected
            // show or hide the advanced options
            if (advanced_LSVM_Options.style.display === 'none') {
                advanced_LSVM_Options.style.display = 'block';
                this.textContent = 'Esconder opções avançadas';
            } else {
                advanced_LSVM_Options.style.display = 'none';
                this.textContent = 'Ver opções avançadas';
            }
            break;
        case '2':
        case '3':
            // 'Função de base radial' or 'Polinomial' is selected
            // show or hide the advanced options
            if (advanced_SVM_Options.style.display === 'none') {
                advanced_SVM_Options.style.display = 'block';
                this.textContent = 'Esconder opções avançadas';
            } else {
                advanced_SVM_Options.style.display = 'none';
                this.textContent = 'Ver opções avançadas';
            }
            break;
        default:
            // Code to execute when no option is selected
            console.log('No kernel selected');
    }
});



/* Show custom gamma option */
function enableInput() {
    document.getElementById('custom-gamma').disabled = false;
    document.getElementById('custom-gamma').style.display = 'block';
}
function disableInput() {
    document.getElementById('custom-gamma').disabled = true;
    document.getElementById('custom-gamma').style.display = 'none';
}







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
    const rows = document.querySelectorAll('tr.clickable-tr');

    rows.forEach(function(row) {
        row.addEventListener('click', function() {
            window.location = this.getAttribute('data-href');
        });
    });
}); */