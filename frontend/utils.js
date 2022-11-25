export function renderTemplate(id) {
    const temp = document.querySelector(`[data-id=${id}]`)
    if (!temp) {
        return console.error(`No Template found for '${id}' `)
    }
    const clon = temp.content.cloneNode(true);
    document.getElementById("content").innerHTML = ""
    document.getElementById("content").appendChild(clon)
}


export async function loadTemplate(page) {
    const resHtml = await fetch(page).then(r => {
        if (!r.ok) {
            throw new Error(`Failed to load the page: '${page}' `)
        }
        return r.text()
    });
    const body = document.getElementsByTagName("BODY")[0];
    const div = document.createElement("div");
    div.innerHTML = resHtml;
    body.appendChild(div)
    return div.querySelector("template")
}


export function setActive(newActive) {
    const linkDivs = document.getElementById("navbar").querySelectorAll("a")
    linkDivs.forEach(div => {
        div.classList.remove("active")
    })
    if (newActive) {
        newActive.classList.add("active")
    }
}


export function encode(str) {
    str = str.replace(/&/g, "&amp;");
    str = str.replace(/>/g, "&gt;");
    str = str.replace(/</g, "&lt;");
    str = str.replace(/"/g, "&quot;");
    str = str.replace(/'/g, "&#039;");
    return str;
}


export async function handleHttpErrors(res) {
    if (!res.ok) {
        const errorResponse = await res.json();
        const error = new Error(errorResponse.message)
        error.apiError = errorResponse
        throw error
    }
    return res.json()
}


export function makeOptions(method, body) {
    const opts = {
        method: method,
        headers: {
            "Content-type": "application/json",
            "Accept": "application/json"
        }
    }
    if (body) {
        opts.body = JSON.stringify(body);
    }
    return opts;
}

export function makeOptionsToken(method, body, addToken) {
    const opts = {
        method: method,
        headers: {
            "Content-type": "application/json",
            "Accept": "application/json"
        }
    }
    if (addToken) {
        opts.headers.Authorization = "Bearer " + sessionStorage.getItem("token")
    }
    if (body) {
        opts.body = JSON.stringify(body);
    }
    return opts;
}

export function showPage(pageId) {
    document.getElementById(pageId).dispatchEvent(clickEvent)
}

const clickEvent = new MouseEvent("click", {
    view: window,
    bubbles: true,
    cancelable: true
});