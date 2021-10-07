--lets fix the column type issue on it.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

--create table for Annual Batting Averages 
CREATE OR REPLACE TABLE annual_bat_avg(
    SELECT
            batter
            , YEAR(g.local_date) AS year
            , ROUND(SUM(Hit)/NULLIF(SUM(atBat),0),3) AS annualBattingAverage
    FROM batter_counts bc 
    JOIN game g ON g.game_id = bc.game_id
    GROUP BY batter, YEAR(g.local_date)
    ORDER BY batter, YEAR(g.local_date)
); 


--create historic batting average 
CREATE OR REPLACE TABLE historic_bat_avg( 
    SELECT
            batter 
            , ROUND(SUM(Hit)/NULLIF(SUM(atBat),0),3) AS historicBattingAverage
    FROM batter_counts 
    GROUP BY batter
);

--create rolling average table 
--including 100_days_prior into table for frame of reference 
--when reading the table
--Note for the earlier dates the BA is inaccurate as it is not truly 100 days
CREATE OR REPLACE TABLE rolling_batting_average(
    SELECT 
            bc.batter
            , g.game_id
            , g.local_date
            , DATE_SUB(g.local_date, INTERVAL 100 DAY) AS 100_days_prior
            , (SELECT ROUND(SUM(bc1.Hit)/NULLIF(SUM(bc1.atBat),0),3)
                FROM batter_counts as bc1
                JOIN game as g1 ON g1.game_id = bc1.game_id
                WHERE 
                    (g1.local_date BETWEEN DATE_SUB(g.local_date, INTERVAL 100 DAY) AND g.local_date) AND
                    (bc1.batter = bc.batter)
                GROUP BY batter) AS 100DayBattingAverage
    FROM batter_counts as bc
    JOIN game as g ON g.game_id = bc.game_id 
    GROUP BY bc.batter, g.game_id
    ORDER BY bc.batter, g.game_id
);

--SHOWING TABLES FOR FIRST BATTER 
SELECT * FROM annual_bat_avg WHERE batter = 110029; 
SELECT * FROM historic_bat_avg WHERE batter = 110029;
SELECT * FROM rolling_batting_average WHERE batter = 110029;
