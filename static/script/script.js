 
 
 function scrollToElement(targetElement) {
    window.scrollTo({
        behavior: 'smooth',
        top: targetElement.offsetTop
    });
}


document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault(); 

        const targetId = this.getAttribute('href').substring(1); 
        const targetElement = document.getElementById(targetId); 

        
        if (targetElement) {
            scrollToElement(targetElement);
        }
    });
});


document.querySelectorAll('button[data-scroll-to]').forEach(button => {
    button.addEventListener('click', function(e) {
        e.preventDefault(); 

        const targetId = this.getAttribute('data-scroll-to');
        const targetElement = document.getElementById(targetId);

        
        if (targetElement) {
            scrollToElement(targetElement);
        }
    });
});


function showUploadBox() {
    const fileInput = document.getElementById("customFile");
    const uploadBox = document.getElementById("uploadBox");
    const fileNameSpan = document.getElementById("fileName");
  
    if (fileInput.value) {
      const fileName = fileInput.value.split("\\").pop(); 
      fileNameSpan.textContent = ` فایل: ${fileName}`;
      uploadBox.classList.remove("d-none"); 
    } else {
      uploadBox.classList.add("d-none"); 
    }
  }


 

  document.getElementById('inputText').addEventListener('click', function() {
    document.getElementById('result').innerHTML = '';

    
});

function submitForm(event) {
    event.preventDefault(); 

    // Get the textarea value
    var textAreaValue = document.getElementById('inputText').value;
    
    // Check if the textarea is empty
    if (!textAreaValue.trim()) {
        showAlert('تکایە ڕستەیەک بنووسە');
        return false;
    }
    
    var kurdishLettersRegex = /^[\u0626\u0627\u0628\u067E\u062A\u062C\u0686\u062D\u062E\u062F\u0631\u0695\u0632\u0698\u0633\u0634\u0639\u063A\u0641\u06A4\u0642\u06A9\u06AF\u0644\u06B5\u0645\u0646\u0648\u06C6\u0647\u06D5\u06CC\u06CE\s.,!?;:'"()]+$/;
        
    var words = textAreaValue.trim().split(/\s+/);

    // Check if the number of words is less than two
    if (words.length < 2) {
        showAlert('تکایە کەمتر لە دوو وشە مەنووسە');
        return false;
    }

    if (!kurdishLettersRegex.test(textAreaValue)) {
        showAlert('تەنها پیتەکانی کوردی بنووسە بەبێ هێماکانی ڕستەسازی.');
        return false;
    }
    

    if (/<script.*?>.*?<\/script>/i.test(textAreaValue)) {
        showAlert('ناتوانیت تاگەکانی جاڤا سکریپت و HTML بنووسیت');
        return false;
    }

    const formData = new FormData(document.getElementById('inputForm'));

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json()) 
    .then(data => {
        let alertClass = '';
        if (data.variable === 'ئەم ڕستە ئەرێنییە✔️') {
            alertClass = 'alert-success';
        } else if (data.variable === 'ئەم ڕستەیە نەرێنییە❌') {
            alertClass = 'alert-danger';
        } else if (data.variable) {
            alertClass = 'alert-secondary';
        }
        


        // Show new result
        showAlert(data.variable, alertClass);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}



function clearTextArea() {
    document.getElementById('inputText').value = ''; // Clear the textarea value
}




// Function to display custom alert
function showAlert(message, alertClass = 'alert-warning') {
    const alertDiv = document.createElement('div');
    alertDiv.classList.add('custom-alert', 'p-3', 'rounded', 'text-dark', alertClass);
    alertDiv.textContent = message;
    document.getElementById('result').innerHTML = '';
    document.getElementById('result').appendChild(alertDiv);
}





function submitFileForm(event) {
    event.preventDefault(); 

    var form = $('#fileForm')[0]; 
    var formData = new FormData(form); 
    
    $.ajax({
        type: 'POST',
        url: '/predict_file',
        data: formData,
        processData: false,
        contentType: false,
        cache: false,
        success: function(response) {
            
            console.log(response); 
        
            
            $('#div').html(response.result);
        
            
            if (response.table) {
               
                $('#table-container').html(response.table);
            }
        
           
            if (response.download_link) {
                var downloadHtml = '<br><hr><div class="text-right mt-3">';
                downloadHtml += '<a href="' + response.download_link + '" class="btn btn-primary" download>';
                downloadHtml += '<i class="bi bi-file-arrow-down"></i> داگرتنی فایل ئەنجامدراو</a>';
                downloadHtml += '</div>';
                $('#download_link').append(downloadHtml);
            }
            
        },
        
        error: function(xhr, status, error) {
            // Handle errors
            console.error(xhr.responseText); 
          
            $('#div').html("هەڵەیەک ڕوویدا، تکایە هەوڵ بدەرەوە."); 
        }
    });
}

function closeToggle() {
    $('.navbar-collapse').collapse('hide');
}


