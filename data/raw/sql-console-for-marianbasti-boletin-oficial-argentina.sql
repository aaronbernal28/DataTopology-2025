/*
Use this query to extract boletin oficial entries related to the executive branch from 2022 to 2025.
Source: https://huggingface.co/datasets/marianbasti/boletin-oficial-argentina/
*/

SELECT 
    row_number() OVER (ORDER BY date DESC) AS id,
    title,
    name,
    full_text,
    date
FROM 
    train
WHERE 
    date BETWEEN '2022-01-01'::DATE AND '2025-12-31'::DATE
    AND entity ILIKE '%PODER EJECUTIVO%'
ORDER BY 
    date DESC