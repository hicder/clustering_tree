//! Attribute column types.

use crate::error::{Result, TreeError};


/// Supported attribute column types in a [`CollectionSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AttributeType {
    String = 1,
    Int = 2,
    Uint = 3,
    Float = 4,
    Bool = 5,
    StringArray = 6,
    IntArray = 7,
    UintArray = 8,
}

impl AttributeType {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            1 => Ok(Self::String),
            2 => Ok(Self::Int),
            3 => Ok(Self::Uint),
            4 => Ok(Self::Float),
            5 => Ok(Self::Bool),
            6 => Ok(Self::StringArray),
            7 => Ok(Self::IntArray),
            8 => Ok(Self::UintArray),
            other => Err(TreeError::CorruptedData(format!(
                "unknown AttributeType byte: {other}"
            ))),
        }
    }

    pub fn is_array(self) -> bool {
        matches!(self, Self::StringArray | Self::IntArray | Self::UintArray)
    }

    pub fn is_ordered(self) -> bool {
        matches!(self, Self::String | Self::Int | Self::Uint | Self::Float)
    }

    /// For array types, returns the scalar element type; for scalar types, returns `self`.
    pub fn scalar_type(self) -> Self {
        match self {
            Self::StringArray => Self::String,
            Self::IntArray => Self::Int,
            Self::UintArray => Self::Uint,
            other => other,
        }
    }
}
