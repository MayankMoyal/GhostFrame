
function updateClock() {

    const now = new Date();

    document.getElementById("date").innerHTML =
        now.toLocaleDateString();

    document.getElementById("time").innerHTML =
        now.toLocaleTimeString();

}

updateClock();

setInterval(updateClock, 1000);


const generateBtn = document.getElementById("generate");
const prompt = document.getElementById("prompt");
const style = document.getElementById("style");



generateBtn.addEventListener("click", function () {

    

    if (prompt.value.trim() === "") {

        alert("Please enter an image prompt.");
        return;

    }



    generateBtn.innerHTML = "Generating...";
    generateBtn.disabled = true;


    document.getElementById("analysis").innerHTML = "Processing...";
    document.getElementById("sceneType").innerHTML = "Detecting...";
    document.getElementById("imageStyle").innerHTML = style.value;
    document.getElementById("lighting").innerHTML = "Analyzing...";

    document.getElementById("step1").innerHTML = "🟢 Prompt Received";
    document.getElementById("step2").innerHTML = "⚪ Prompt Enhancement";
    document.getElementById("step3").innerHTML = "⚪ AI Processing";
    document.getElementById("step4").innerHTML = "⚪ Image Rendering";
    document.getElementById("step5").innerHTML = "⚪ Ready";



    setTimeout(function () {

        document.getElementById("step2").innerHTML =
            "🟢 Prompt Enhancement";

    }, 1000);

    setTimeout(function () {

        document.getElementById("step3").innerHTML =
            "🟢 AI Processing";

    }, 2000);

    setTimeout(function () {

        document.getElementById("step4").innerHTML =
            "🟢 Image Rendering";

    }, 3500);

    setTimeout(function () {

        document.getElementById("step5").innerHTML =
            "🟢 Ready";

        document.getElementById("analysis").innerHTML =
            "Completed";

        document.getElementById("sceneType").innerHTML =
            "Landscape";

        document.getElementById("lighting").innerHTML =
            "Soft Natural Light";


        generateBtn.innerHTML = "✨ Generate Image";
        generateBtn.disabled = false;

    }, 5000);

});