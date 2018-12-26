package com.example.ian.titanscout

import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import org.json.JSONObject



// Classes taken from https://stackoverflow.com/questions/41928803/how-to-parse-json-in-kotlin.
// Inputs a string and outputs a JSONObject. This was a quick alternative to importing another library.
class Response(json: String) : JSONObject(json) {
    val type: String? = this.optString("type")
    val data = this.optJSONArray("data")
        ?.let { 0.until(it.length()).map { i -> it.optJSONObject(i) } } // returns an array of JSONObject
        ?.map { Foo(it.toString()) } // transforms each JSONObject of the array into Foo
}
// Helper for the above class
class Foo(json: String) : JSONObject(json) {
    val id = this.optInt("id")
    val title: String? = this.optString("title")
}


class PrematchesActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_alias)



        findViewById<Button>(R.id.updateProfileButton).setOnClickListener {


            // Get alias from the edittext and clear the edittext
            val alias = findViewById<EditText>(R.id.aliasField).text.toString()
            findViewById<EditText>(R.id.aliasField).text.clear()

            //
            val intent2 = Intent(this, MatchesTableView::class.java)
            intent2.putExtra("alias", alias)
            intent2.putExtra("auth", intent.getStringExtra("auth"))
            startActivity(intent2)


        }

    }

}


