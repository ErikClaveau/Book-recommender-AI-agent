from pydantic import BaseModel, Field
from typing import List

"""
Data models for book recommendations.

Defines the Book and RecommendedBooks schemas used for structured outputs
from the LLM. These models ensure consistent data extraction and formatting.
"""


class Book(BaseModel):
    """
    Represents a single book with its title and author.

    Attributes:
        name (str): The name (title) of the book.
        author (str): The book's author name.
    """
    name: str = Field(..., description="The name of the book")
    author: str = Field(..., description="The book's author name")

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the book.

        Returns:
            str: A string in the format "<name> — <author>".
        """
        return f"{self.name} — {self.author}"


class RecommendedBooks(BaseModel):
    """
    Schema for a list of book recommendations produced by the LLM.

    Attributes:
        recommended_books (List[Book]): List of Book instances recommended by the model.
    """
    recommended_books: List[Book] = Field(..., description="The different recommended books")

    def __str__(self) -> str:
        """
        Return a newline-separated string of all recommended books.

        Returns:
            str: A formatted list of books, or an empty string if none.
        """
        if self.recommended_books:
            return "\n".join(str(book) for book in self.recommended_books)
        return ""
