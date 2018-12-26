package com.example.ian.titanscout


class Match (var ind: Int, var redTeams: Array<Team>, var blueTeams: Array<Team>) {
    // class body

    fun getScouts(): Int {
        var x = 0
        for (red in redTeams) {
            if (red.hasScouts()) {
                x++
            }
        }
        for (blue in blueTeams) {
            if (blue.hasScouts()) {
                x++
            }
        }
        return x
    }
}

class Team (var num: String, var color:String, var scouts:Array<String>) {

    fun hasScouts() : Boolean {
        if (scouts.size > 0 && !scouts[0].equals("")) {
            return true
        }
        return false
    }
}