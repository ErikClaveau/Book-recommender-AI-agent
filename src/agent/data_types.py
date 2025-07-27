"""
Data models for book recommendations and user preferences.

Defines the Book, RecommendedBooks, and Preferences schemas used for structured
outputs from the LLM. These models ensure consistent data extraction,
validation, and formatting throughout the recommendation workflow.
"""
from pydantic import BaseModel, Field
from typing import List


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
        recommended_books (List[Book]):
            List of Book instances recommended by the model.
    """
    recommended_books: List[Book] = Field(
        ..., description="The different recommended books"
    )

    def __str__(self) -> str:
        """
        Return a newline-separated string of all recommended books.

        Returns:
            str: A formatted list of books, or an empty string if none.
        """
        if self.recommended_books:
            return "\n".join(str(book) for book in self.recommended_books)
        return ""


class Preferences(BaseModel):
    """
    Schema for capturing user reading preferences or genres.

    Attributes:
        preferences (List[str]):
            A list of user-specified reading preferences.
    """
    preferences: List[str] = Field(
        ..., description="The different preferences the user has"
    )

    def __str__(self) -> str:
        """
        Return a newline-separated string of all user preferences.

        Returns:
            str: A formatted list of preferences, or an empty string if none.
        """
        if self.preferences:
            return "\n".join(self.preferences)
        return ""
