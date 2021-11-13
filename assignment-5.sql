-- lets fix the column type issue on it.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

-- create historic batting average, BABIP, OBP 
CREATE TEMPORARY TABLE historic_bat_avg( 
    SELECT
            batter
            , ROUND(SUM(Hit)/NULLIF(SUM(atBat),0),3) AS historicBattingAverage
            , ROUND(SUM(Hit - Home_Run)/NULLIF(SUM(atBat - Strikeout - Home_Run + Sac_Fly),0),3) AS historicBABIP
            , ROUND(SUM(Hit + Walk + Home_Run)/NULLIF(SUM(atBat + Walk + Hit_By_Pitch + Sac_Fly),0),3) AS OBP
    FROM batter_counts
    GROUP BY batter
);

-- create table for avg of lineup
CREATE TEMPORARY TABLE line_up_temp(
    SELECT
            l.game_id
            ,l.team_id
            , AVG(historicBattingAverage) AS BA
            , AVG(historicBABIP) AS BABIP
            , AVG(OBP) AS OBP
    FROM historic_bat_avg hba
        JOIN lineup l ON hba.batter = l.player_id
    GROUP BY l.team_id, l.game_id
);

-- Find League stats 
CREATE TEMPORARY TABLE league_stats(
    SELECT 
        t.league
        , ROUND(((SUM(toBase)/SUM(endingInning - startingInning + 1)) * 9),3) AS lgERA
        , SUM(Home_Run) AS lgHR
        , SUM(Walk) AS lgBB
        , SUM(Strikeout) AS lgK
        , SUM(endingInning - startingInning + 1) AS lgIP
        , SUM((Fly_Out + Flyout)/NULLIF(Home_Run,0)) AS lgHRFB
    FROM pitcher_counts p
        JOIN team t ON p.team_id = t.team_id
    GROUP BY t.league
);

CREATE TEMPORARY TABLE FIP_constants(
    SELECT league
        ,(lgERA - ((13*lgHR + 3*lgBB - 2*lgK)/lgIP)) AS C
        , lgHRFB
    FROM league_stats
); 

-- Make Junction table
CREATE TEMPORARY TABLE FIP_constants_junt(
    SELECT f.league
        ,t.team_id
        , C
        , lgHRFB
    FROM FIP_constants f
        JOIN team t ON f.league = t.league
);

-- table for pitcher stats K/9, BB/9, FIP, xFIP, ERA
CREATE TEMPORARY TABLE pitcher_temp(
    SELECT
        p.pitcher
        , ROUND(((SUM(p.Strikeout)/SUM(p.endingInning - p.startingInning + 1)) * 9),3) AS K9
        , ROUND(((SUM(p.Walk)/SUM(p.endingInning - p.startingInning + 1)) * 9),3) AS BB9
        , ROUND((((13*SUM(p.Home_Run) + 3*SUM(p.Walk) - 2*SUM(p.Strikeout))/SUM(p.endingInning - p.startingInning + 1)) + f.C),3) AS FIP
        , ROUND((((13*SUM(p.Fly_Out + p.Flyout)*f.lgHRFB + 3*SUM(p.Walk) - 2*SUM(p.Strikeout))/SUM(p.endingInning - p.startingInning + 1)) + f.C),3) AS xFIP
        , ROUND(((SUM(p.toBase)/SUM(p.endingInning - p.startingInning + 1)) * 9),3) AS ERA
    FROM pitcher_counts p
        JOIN FIP_constants_junt f ON p.team_id = f.team_id
    GROUP BY p.pitcher
);


CREATE TEMPORARY TABLE results_table(
    SELECT
        game_id
        , CASE WHEN win_lose="L" THEN 'FALSE' ELSE 'TRUE' END AS result
    FROM team_results
    WHERE home_away = "H"
);

-- home team stats table
CREATE TEMPORARY TABLE home_team_stats(
    SELECT
        g.game_id
        , l.team_id
        , l.BA as home_BA
        , l.BABIP as home_BABIP
        , l.OBP as home_OBP
        , p.K9 as home_K9
        , p.BB9 as home_BB9
        , p.FIP as home_FIP
        , p.xFIP as home_xFIP
        , p.ERA as home_ERA

    FROM game g
        JOIN line_up_temp l ON g.game_id = l.game_id AND g.home_team_id = l.team_id
        JOIN pitcher_temp p ON g.home_pitcher = p.pitcher
);

-- away team stats table
CREATE TEMPORARY TABLE away_team_stats(
    SELECT
        g.game_id
        , l.team_id
        , l.BA as away_BA
        , l.BABIP as away_BABIP
        , l.OBP as away_OBP
        , p.K9 as away_K9
        , p.BB9 as away_BB9
        , p.FIP as away_FIP
        , p.xFIP as away_xFIP
        , p.ERA as away_ERA

    FROM game g
        JOIN line_up_temp l ON g.game_id = l.game_id AND g.away_team_id = l.team_id
        JOIN pitcher_temp p ON g.away_pitcher = p.pitcher
);

-- Join all the stats i made into one table plus other features 
CREATE OR REPLACE TABLE predictive_table(
    SELECT
        g.game_id
        , g.home_team_id
        , g.away_team_id
        , a.away_BA
        , a.away_BABIP
        , a.away_OBP
        , a.away_K9
        , a.away_BB9
        , a.away_FIP
        , a.away_xFIP
        , a.away_ERA
        , h.home_BA
        , h.home_BABIP
        , h.home_OBP
        , h.home_K9
        , h.home_BB9
        , h.home_FIP
        , h.home_xFIP
        , h.home_ERA
        , g.home_pitcher
        , g.away_pitcher
        , r.result
    FROM game g
        JOIN away_team_stats a ON g.game_id = a.game_id AND g.away_team_id = a.team_id
        JOIN home_team_stats h ON g.game_id = h.game_id AND g.home_team_id = h.team_id
        JOIN results_table r ON g.game_id = r.game_id
);