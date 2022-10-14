# ADR1_Database: Use of SQL over Non-relational database

## Overview
Every modern web application leverages one or more databases to store information. Databases provide ways of defining your data structures,
inserting new data, finding existing data, updating or deleting existing data, performing computations across the data, and more.

SQL(Structured Query Language) uses relatioal database. This means that data is stored in a table format. 
Non-relational databases stores data in the form of documents.

## Decision
We will use SQL for out database with Microsoft SQL Studio Management to view changes locally

## Status
Accepted

## Consequences

      ## Advantages
      Large amount of data is retrieved quickly and efficiently. Operations like Insertion, deletion, manipulation of data is also done in almost no time.

      For data retrieval, large number of lines of code is not required. All basic keywords such as SELECT, INSERT INTO, UPDATE, etc are used and also the
      syntactical rules are not complex in SQL, which makes it a user-friendly language.

     Easy to learn and understand, answers to complex queries can be received quickly.

     SQL can be used in the program in PCs, servers, laptops, and even some of the mobile phones.

     ## Disadvantages
     SQL has a complex interface that makes it difficult for some users to access it.

     The programmers who use SQL doesn’t have a full control over the database because of the hidden business rules.

     The operating cost of some SQL versions makes it difficult for some programmers to access it.