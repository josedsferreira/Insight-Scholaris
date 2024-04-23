
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


/* Show or hide Kernel selector Form */
var algoTypeElement = document.getElementById('algo-type');
if (algoTypeElement) {
    algoTypeElement.addEventListener('change', function() {
        var selectedAlgo = this.value;
        const kernelForm = document.getElementById('kernel-form');
        const selector = document.getElementById('kernel');

        switch(selectedAlgo) {
            case '1':
                // SVM is selected
                if (kernelForm.style.display === 'none') {
                    kernelForm.style.display = 'block';
                }
                if (selector.hasAttribute('disabled')) {
                    selector.removeAttribute('disabled');
                }
                break;
            case '2':
                // XGB is selected
                if (kernelForm.style.display === 'block') {
                    kernelForm.style.display = 'none';
                }
                if (!selector.hasAttribute('disabled')) {
                    selector.setAttribute('disabled', 'disabled');
                }
                break;
            case '3':
                // RF is selected
                if (kernelForm.style.display === 'block') {
                    kernelForm.style.display = 'none';
                }
                if (!selector.hasAttribute('disabled')) {
                    selector.setAttribute('disabled', 'disabled');
                }
                break;
            default:
                // Code to execute when no option is selected
                console.log('No algorithm selected');
        }
    });
}


/* show advanced options button and the basic options only when kernel is selected */
/* document.getElementById('kernel').addEventListener('change', function() {

    const selectedOption = this.value;

    // get toggle button
    const button = document.getElementById('toggle-advanced-svm-options');

    // get the basic options
    const basic_SVM_Options = document.getElementById('basic-svm-options');
    const basic_LSVM_Options = document.getElementById('basic-lsvm-options');

    // get the advanced options div, they should be hidden when changing kernel
    const advanced_poli_SVM_Options = document.getElementById('advanced-poli-svm-options');
    const advanced_rbf_SVM_Options = document.getElementById('advanced-rbf-svm-options');
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
    if (advanced_poli_SVM_Options.style.display === 'block') {
        advanced_poli_SVM_Options.style.display = 'none';
    }
    if (advanced_rbf_SVM_Options.style.display === 'block') {
        advanced_rbf_SVM_Options.style.display = 'none';
    }
    if (advanced_LSVM_Options.style.display === 'block') {
        advanced_LSVM_Options.style.display = 'none';
    }
}); */


/* Toggle advanced SVM options */
toggleAdvancedSVM = document.getElementById('toggle-advanced-svm-options');
if (toggleAdvancedSVM) {
    toggleAdvancedSVM.addEventListener('click', function() {
    
        // get the options div
        const advanced_Options = document.getElementById('advanced-svm-options');

        // Select all form elements inside the div
        var formElements = advanced_Options.querySelectorAll('input, select, textarea, button');

        // show or hide the advanced options
        if (advanced_Options.style.display === 'none') {
            advanced_Options.style.display = 'block';
            this.textContent = 'Esconder opções avançadas';

            // Loop through the form elements and turn them on
            formElements.forEach(function(element) {
                if (element.id !== 'max-iter-custom') {
                    element.disabled = false;
                }
            });

        } else {
            advanced_Options.style.display = 'none';
            this.textContent = 'Ver opções avançadas';

            // Loop through the form elements and disable them
            formElements.forEach(function(element) {
                element.disabled = true;
            });
        }
    });
}


/* Toggle advanced XGB options */
toggleAdvancedXGB = document.getElementById('toggle-advanced-xgb-options');
if (toggleAdvancedXGB) {
    toggleAdvancedXGB.addEventListener('click', function() {
    
        // get the options div
        const advanced_Options = document.getElementById('advanced-XGB-options');

        // Select all form elements inside the div
        var formElements = advanced_Options.querySelectorAll('input, select, textarea, button');

        // show or hide the advanced options
        if (advanced_Options.style.display === 'none') {
            advanced_Options.style.display = 'block';
            this.textContent = 'Esconder opções avançadas';

            // Loop through the form elements and turn them on
            formElements.forEach(function(element) {
                element.disabled = false;
            });

        } else {
            advanced_Options.style.display = 'none';
            this.textContent = 'Ver opções avançadas';

            // Loop through the form elements and disable them
            formElements.forEach(function(element) {
                element.disabled = true;
            });
        }
    });
}


/* Show custom gamma option */
function enableGammaInput() {
    document.getElementById('custom-gamma').disabled = false;
    document.getElementById('custom-gamma').style.display = 'block';
}
function disableGammaInput() {
    document.getElementById('custom-gamma').disabled = true;
    document.getElementById('custom-gamma').style.display = 'none';
}


/* Show custom max iter option */
function enableMaxIterInput() {
    document.getElementById('max-iter-custom').disabled = false;
    document.getElementById('max-iter-custom').style.display = 'block';
}
function disableMaxIterInput() {
    document.getElementById('max-iter-custom').disabled = true;
    document.getElementById('max-iter-custom').style.display = 'none';
}


/* Show or hide custom random_state */
var randomStateElements = document.getElementsByName('random_state');
var randomStateValueElement = document.getElementById('random-state-value');
if (randomStateElements.length > 0 && randomStateValueElement) {
    randomStateElements.forEach(function(elm) {
        elm.addEventListener('change', function() {
            randomStateValueElement.style.display = (this.value == 'set') ? 'block' : 'none';
            randomStateValueElement.disabled = (this.value == 'set') ? false : true;
        });
    });
}


/* Hide or show Random State div */
var probabilityElements = document.getElementsByName('probability');
var randomStateDiv = document.getElementById('random-state-div');
if (probabilityElements.length > 0 && randomStateDiv) {
    probabilityElements.forEach(function(elm) {
        elm.addEventListener('change', function() {
            randomStateDiv.style.display = (this.value == 'true') ? 'block' : 'none';
            element.disabled = (this.value == 'true') ? false : true;
        });
    });
}


/* Submit Form with split and dataset */
function submitSplitAndDSForm(formId) {
    // Get the value of the split input
    var split = document.getElementById('split').value;

    // Set the value of the hidden split input in the form
    document.getElementById('hiddenSplit' + formId).value = split;

    // Submit the form
    document.getElementById('form' + formId).submit();
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