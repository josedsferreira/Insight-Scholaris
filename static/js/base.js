const body = document.querySelector('body'),
    sidebar = document.querySelector('.sidebar'),
    toggle = document.querySelector('.toggle');

sidebar.addEventListener('mouseover', () => {
        sidebar.classList.remove("close");
    });
    
sidebar.addEventListener('mouseout', () => {
        sidebar.classList.add("close");
    });

$(document).ready(function(){
        $("tr.clickable-tr").click(function(){
            window.location = $(this).data('href');
        });
    });