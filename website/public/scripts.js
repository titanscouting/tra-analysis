// Initialize Firebase
var config = {
  apiKey: "(insert the TitanScout Api Key Here)",
  authDomain: "titanscoutandroid.firebaseapp.com",
  databaseURL: "https://titanscoutandroid.firebaseio.com",
  projectId: "titanscoutandroid",
  storageBucket: "titanscoutandroid.appspot.com",
  messagingSenderId: "1097635313476"
};
firebase.initializeApp(config);
// FirebaseUI config.
var uiConfig = {
  signInSuccessUrl: '<url-to-redirect-to-on-success>',
  signInOptions: [
    // Leave the lines as is for the providers you want to offer your users.
    firebase.auth.GoogleAuthProvider.PROVIDER_ID,
    //firebase.auth.FacebookAuthProvider.PROVIDER_ID,
    //firebase.auth.TwitterAuthProvider.PROVIDER_ID,
    firebase.auth.GithubAuthProvider.PROVIDER_ID,
    firebase.auth.EmailAuthProvider.PROVIDER_ID,
    firebase.auth.PhoneAuthProvider.PROVIDER_ID,
    //  firebaseui.auth.AnonymousAuthProvider.PROVIDER_ID
  ],
  // tosUrl and privacyPolicyUrl accept either url string or a callback
  // function.
  // Terms of service url/callback.
  tosUrl: function() {
    alert("this is a test app. don't use it");
  },
  // Privacy policy url/callback.
  privacyPolicyUrl: function() {
    alert("we will steal all of the data");
  }
};
// Initialize the FirebaseUI Widget using Firebase.
var ui = new firebaseui.auth.AuthUI(firebase.auth());

// The start method will wait until the DOM is loaded.
ui.start('#firebaseui-auth-container', uiConfig);
