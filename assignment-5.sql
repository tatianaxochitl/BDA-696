-- lets fix the column type issue on it.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

-- create for every entry batting average, BABIP, OBP 
CREATE TABLE IF NOT EXISTS bat_avg1( 
    SELECT
            game_id
            , team_id
            , Hit/NULLIF(atBat,0) AS BattingAverage
            , (Hit - Home_Run)/NULLIF((atBat - Strikeout - Home_Run + Sac_Fly),0) AS BABIP
            , (Hit + Walk + Home_Run)/NULLIF((atBat + Walk + Hit_By_Pitch + Sac_Fly),0) AS OBP
    FROM batter_counts
    GROUP BY game_id, team_id
    ORDER BY game_id, team_id
);


CREATE TABLE IF NOT EXISTS bat_avg2( 
    SELECT
            game_id
            , team_id
            , AVG(BattingAverage) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING ) AS BA
            , AVG(BABIP) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING ) AS BABIP
            , AVG(OBP) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING ) AS OBP
    FROM bat_avg1 
    ORDER BY game_id, team_id
);

-- Find League stats 
-- Doing Historic cuz im lazy... 
-- Also not exactly accurate way to get ERA but gives similar number :-)
CREATE TABLE IF NOT EXISTS league_stats(
    SELECT 
        t.league
        , ((SUM((atBat - outsPlayed)/2)/SUM(endingInning - startingInning + 1)) * 9) AS lgERA
        , SUM(Home_Run) AS lgHR
        , SUM(Walk) AS lgBB
        , SUM(Strikeout) AS lgK
        , SUM(endingInning - startingInning + 1) AS lgIP
        , AVG(COALESCE((Home_Run/NULLIF((Fly_Out + Flyout),0)),0)) AS lgHRFB
    FROM pitcher_counts p
        JOIN team t ON p.team_id = t.team_id
    GROUP BY t.league
);

CREATE TABLE IF NOT EXISTS FIP_constants(
    SELECT league
        ,(lgERA - ((13*lgHR + 3*lgBB - 2*lgK)/lgIP)) AS C
        , lgHRFB
    FROM league_stats
); 

-- Make Junction table
CREATE TABLE IF NOT EXISTS FIP_constants_junt(
    SELECT f.league
        , t.division
        ,t.team_id
        , C
        , lgHRFB
    FROM FIP_constants f
        JOIN team t ON f.league = t.league
);

-- table for pitcher stats K/9, BB/9, FIP, xFIP, ERA, WHIP
CREATE TABLE IF NOT EXISTS pitcher_temp1(
    SELECT
        p.game_id
        , p.team_id
        , p.pitcher
        , p.endingInning - p.startingInning + 1 AS IP
        , ((p.Strikeout/(p.endingInning - p.startingInning + 1)) * 9) AS K9
        , ((p.Walk/(p.endingInning - p.startingInning + 1)) * 9) AS BB9
        , (((13*(p.Home_Run) + 3*(p.Walk) - 2*(p.Strikeout))/(p.endingInning - p.startingInning + 1)) + f.C) AS FIP
        , (((13*(p.Fly_Out + p.Flyout)*f.lgHRFB + 3*(p.Walk) - 2*(p.Strikeout))/(p.endingInning - p.startingInning + 1)) + f.C) AS xFIP
        , ((((p.atBat - p.outsPlayed)/2)/(p.endingInning - p.startingInning + 1)) * 9) AS ERA
        , (p.Walk + p.Hit) / (p.endingInning - p.startingInning + 1) AS WHIP
        , f.league
        , f.division
    FROM pitcher_counts p
        JOIN FIP_constants_junt f ON p.team_id = f.team_id
    ORDER BY p.game_id, p.pitcher
);

CREATE TABLE IF NOT EXISTS pitcher_temp2(
    SELECT
        game_id
        , team_id
        , SUM(K9 * IP)/SUM(IP) AS K9
        , SUM(BB9 * IP)/SUM(IP) AS BB9
        , SUM(FIP * IP)/SUM(IP) AS FIP
        , SUM(xFIP * IP)/SUM(IP) AS xFIP
        , SUM(ERA * IP)/SUM(IP) AS ERA
        , SUM(WHIP * IP)/SUM(IP) AS WHIP
        , league
        , division
    FROM pitcher_temp1
    GROUP BY team_id, game_id
    ORDER BY game_id, team_id
);

CREATE TABLE IF NOT EXISTS pitcher_temp3(
    SELECT
        game_id
        , team_id
        , AVG(K9) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING ) AS K9
        , AVG(BB9) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING ) AS BB9
        , AVG(FIP) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING ) AS FIP
        , AVG(xFIP) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING ) AS xFIP
        , AVG(ERA) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING ) AS ERA
        , AVG(WHIP) OVER(PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING ) AS WHIP
        , league
        , division 
    FROM pitcher_temp2
    ORDER BY game_id, team_id
);


CREATE TABLE IF NOT EXISTS results_table(
    SELECT
        game_id
        , CASE WHEN win_lose="L" THEN 'FALSE' ELSE 'TRUE' END AS result
    FROM team_results
    WHERE home_away = "H"
);

-- home team stats table
CREATE TABLE IF NOT EXISTS home_team_stats(
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
        , p.WHIP as home_WHIP
        , p.league as home_league
        , p.division as home_division

    FROM game g
        JOIN bat_avg2 l ON g.game_id = l.game_id AND g.home_team_id = l.team_id
        JOIN pitcher_temp3 p ON g.game_id = p.game_id AND g.home_team_id = p.team_id
);

-- away team stats table
CREATE TABLE IF NOT EXISTS away_team_stats(
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
        , p.WHIP as away_WHIP
        , p.league as away_league
        , p.division as away_division

    FROM game g
        JOIN bat_avg2 l ON g.game_id = l.game_id AND g.away_team_id = l.team_id
        JOIN pitcher_temp3 p ON g.game_id = p.game_id AND g.away_team_id = p.team_id
);

-- Join all the stats i made into one table plus other features 
CREATE TABLE IF NOT EXISTS predictive_table(
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
        , a.away_WHIP
        , a.away_league
        , a.away_division
        , h.home_BA
        , h.home_BABIP
        , h.home_OBP
        , h.home_K9
        , h.home_BB9
        , h.home_FIP
        , h.home_xFIP
        , h.home_ERA
        , h.home_WHIP
        , h.home_league
        , h.home_division
        , r.result
    FROM game g
        JOIN away_team_stats a ON g.game_id = a.game_id AND g.away_team_id = a.team_id
        JOIN home_team_stats h ON g.game_id = h.game_id AND g.home_team_id = h.team_id
        JOIN results_table r ON g.game_id = r.game_id
);