const body = document.querySelector('body'),
    sidebar = document.querySelector('.sidebar'),
    toggle = document.querySelector('.toggle');

    /* toggle.addEventListener('click', () => {
        sidebar.classList.toggle("close");
    }); */

    sidebar.addEventListener('mouseover', () => {
        sidebar.classList.remove("close");
    });
    
    sidebar.addEventListener('mouseout', () => {
        sidebar.classList.add("close");
    });