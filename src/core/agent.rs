use bevy::prelude::Resource;
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlayerAction {
    #[default]
    Up,
    Down,
    Left,
    Right,
}

impl PlayerAction {
    pub const COUNT: usize = 4;
}

impl From<usize> for PlayerAction {
    fn from(value: usize) -> Self {
        match value {
            0 => PlayerAction::Up,
            1 => PlayerAction::Down,
            2 => PlayerAction::Left,
            3 => PlayerAction::Right,
            _ => panic!("Invalid action index"),
        }
    }
}

impl From<PlayerAction> for usize {
    fn from(action: PlayerAction) -> Self {
        match action {
            PlayerAction::Up => 0,
            PlayerAction::Down => 1,
            PlayerAction::Left => 2,
            PlayerAction::Right => 3,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ActionSpace {
    /// Represents a discrete action space with a specific number of actions.
    /// The actions are typically indexed from 0 to n-1.
    Discrete(usize),
}

impl ActionSpace {
    pub fn n_discrete(&self) -> Option<usize> {
        match self {
            ActionSpace::Discrete(n) => Some(*n),
        }
    }
    pub fn sample_index(&self) -> usize {
        match self {
            ActionSpace::Discrete(n) => {
                if *n == 0 {
                    panic!("Cannot sample from a discrete action space with 0 actions.");
                }
                let mut rng = rand::rng();
                rng.random_range(0..*n)
            }
        }
    }
}
