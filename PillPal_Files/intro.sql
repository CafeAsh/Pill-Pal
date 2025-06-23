/* CREATE TABLE pills(
    id integer primary key,
    name TEXT not null,
    manufacturer TEXT,
    color1 TEXT not null,
    color2 TEXT,
    shape TEXT not null,
    size TEXT not null,
    created_at datetime default current_timestamp
); */

/* ALTER table pills add column status text; */

/* drop table users; */

/* insert into pills (name, color1, shape, size)
Values
('Allertec', 'white', 'oblong', 'small'),
('Iron', 'red', 'disk', 'medium'),
('Caffeine', 'white', 'capsule', 'large'),
('Cordyceps', 'beige', 'capsule', 'large'),
('MultiVitamin', 'beige', 'oblong', 'large'),
('Zinc', 'white', 'oblong', 'medium'),
('Aleve', 'blue', 'oblong', 'medium'); */

/* SELECT * FROM pills */

/* delete from pills where id between 8 and 14; */

insert into pills (name, color1, color2, shape, size)
Values
('Tylenol', 'red', 'blue', 'capsule', 'medium');