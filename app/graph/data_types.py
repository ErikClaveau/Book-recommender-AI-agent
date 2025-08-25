"""
Data models for book recommendations, user preferences, and reading history.

Defines the Book, RecommendedBooks, Preferences, and ReadBooks schemas used for structured
outputs from the LLM. These models ensure consistent data extraction,
validation, and formatting throughout the recommendation workflow.
"""
from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class IntentEnum(str, Enum):
    """
    Enumeration of possible intent classifications.
    """
    RECOMMENDATION = "recommendation"
    TALK = "talk"
    PREFERENCES = "preferences"
    READ = "read"
    END = "end"


class IntentClassification(BaseModel):
    """
    Schema for intent classification results.

    Attributes:
        intents (List[IntentEnum]): List of detected intents for routing.
    """
    intents: List[IntentEnum] = Field(
        ..., description="List of detected intent categories"
    )


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
        if not self.recommended_books:
            return ""
        return "\n".join(str(book) for book in self.recommended_books)


class Preferences(BaseModel):
    """
    Schema for user reading preferences extracted from conversations.

    Attributes:
        preferences (List[str]): List of user's reading preferences.
    """
    preferences: List[str] = Field(
        ..., description="List of user reading preferences"
    )

    def __str__(self) -> str:
        """
        Return a formatted string of user preferences.

        Returns:
            str: A formatted list of preferences, or an empty string if none.
        """
        if not self.preferences:
            return ""
        return "\n".join(f"- {pref}" for pref in self.preferences)


class ReadBooks(BaseModel):
    """
    Schema for books that the user has read.

    Attributes:
        read_books (List[Book]): List of books the user has read.
    """
    read_books: List[Book] = Field(
        ..., description="List of books the user has read"
    )

    def __str__(self) -> str:
        """
        Return a formatted string of read books.

        Returns:
            str: A formatted list of read books, or an empty string if none.
        """
        if not self.read_books:
            return ""
        return "\n".join(str(book) for book in self.read_books)
