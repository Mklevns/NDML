#!/usr/bin/env python3
"""Training script with NDML integration"""

import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import wandb
from datetime import datetime

from ndml.integration.llm_wrapper import NDMLIntegratedLLM
from ndml.utils.metrics import NDMLMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NDMLTrainer:
    """Trainer for LLMs with NDML integration"""

    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = self.initialize_model()

        # Initialize metrics
        self.metrics = NDMLMetrics()

        # Initialize wandb
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project'],
                name=self.config['logging']['run_name'],
                config=self.config
            )

    def load_config(self, config_path):
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_model(self):
        """Initialize NDML-integrated model"""
        logger.info("Initializing NDML-integrated model...")

        model = NDMLIntegratedLLM(
            model_name_or_path=self.config['model']['name'],
            memory_dimension=self.config['memory']['dimension'],
            memory_config=self.config['memory']
        )

        model.to(self.device)

        # Enable gradient checkpointing if configured
        if self.config['training'].get('gradient_checkpointing', False):
            model.base_model.gradient_checkpointing_enable()

        return model

    def prepare_data(self):
        """Prepare training data"""
        from datasets import load_dataset

        logger.info("Loading dataset...")

        # Load dataset
        dataset = load_dataset(
            self.config['data']['dataset_name'],
            self.config['data'].get('dataset_config')
        )

        # Tokenize dataset
        tokenizer = self.model.tokenizer

        def tokenize_function(examples):
            return tokenizer(
                examples[self.config['data']['text_column']],
                padding='max_length',
                truncation=True,
                max_length=self.config['data']['max_length']
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            tokenized_dataset['train'],
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )

        eval_dataloader = DataLoader(
            tokenized_dataset['validation'],
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            num_workers=4
        )

        return train_dataloader, eval_dataloader

    def setup_optimization(self, num_training_steps):
        """Setup optimizer and scheduler"""
        # Separate parameters for base model and memory system
        base_params = []
        memory_params = []

        for name, param in self.model.named_parameters():
            if 'memory_gateway' in name or 'fusion_network' in name:
                memory_params.append(param)
            else:
                base_params.append(param)

        # Different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.config['training']['learning_rate']},
            {'params': memory_params, 'lr': self.config['training']['memory_learning_rate']}
        ], weight_decay=self.config['training']['weight_decay'])

        # Linear schedule with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def train_epoch(self, epoch, train_dataloader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with memory
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                context={'epoch': epoch, 'step': step},
                update_memory=True
            )

            # Compute loss
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = batch['input_ids'][..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Add memory regularization loss
            if self.config['training'].get('memory_regularization', 0.0) > 0:
                memory_reg_loss = self.compute_memory_regularization()
                loss = loss + self.config['training']['memory_regularization'] * memory_reg_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            self.metrics.update('train_loss', loss.item())

            # Log to wandb
            if self.config['logging']['use_wandb'] and step % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'memory_stats': self.model.get_memory_stats()
                })

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Periodic memory consolidation
            if step % self.config['memory']['consolidation_interval'] == 0:
                asyncio.create_task(
                    self.model.memory_gateway.periodic_maintenance()
                )

        return total_loss / len(train_dataloader)

    def evaluate(self, eval_dataloader):
        """Evaluate model"""
        self.model.eval()

        total_loss = 0
        total_perplexity = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    update_memory=False  # Don't update memory during evaluation
                )

                # Compute loss
                shift_logits = outputs['logits'][..., :-1, :].contiguous()
                shift_labels = batch['input_ids'][..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                total_loss += loss.item()
                total_perplexity += torch.exp(loss).item()

        avg_loss = total_loss / len(eval_dataloader)
        avg_perplexity = total_perplexity / len(eval_dataloader)

        return avg_loss, avg_perplexity

    def compute_memory_regularization(self):
        """Compute memory regularization loss"""
        # Encourage diversity in memory representations
        memory_stats = self.model.get_memory_stats()

        # Simple regularization: penalize very high or very low utilization
        utilization = memory_stats['system_summary']['average_utilization']
        target_utilization = 0.7

        reg_loss = (utilization - target_utilization) ** 2

        return torch.tensor(reg_loss, device=self.device)

    def save_checkpoint(self, epoch, optimizer, scheduler, best_loss):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(
            self.config['training']['output_dir'],
            f'checkpoint-epoch-{epoch}'
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.base_model.save_pretrained(checkpoint_dir)
        self.model.tokenizer.save_pretrained(checkpoint_dir)

        # Save memory system
        memory_checkpoint_path = os.path.join(
            checkpoint_dir,
            'memory_checkpoint.pt'
        )
        self.model.save_memory_checkpoint(memory_checkpoint_path)

        # Save training state
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'config': self.config
        }, os.path.join(checkpoint_dir, 'training_state.pt'))

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        # Prepare data
        train_dataloader, eval_dataloader = self.prepare_data()

        # Setup optimization
        num_training_steps = len(train_dataloader) * self.config['training']['num_epochs']
        optimizer, scheduler = self.setup_optimization(num_training_steps)

        # Training loop
        best_loss = float('inf')

        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"Starting epoch {epoch + 1}/{self.config['training']['num_epochs']}")

            # Train
            train_loss = self.train_epoch(epoch, train_dataloader, optimizer, scheduler)
            logger.info(f"Epoch {epoch + 1} - Train loss: {train_loss:.4f}")

            # Evaluate
            eval_loss, eval_perplexity = self.evaluate(eval_dataloader)
            logger.info(f"Epoch {epoch + 1} - Eval loss: {eval_loss:.4f}, Perplexity: {eval_perplexity:.2f}")

            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'eval_perplexity': eval_perplexity
                })

            # Save checkpoint if best
            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save_checkpoint(epoch + 1, optimizer, scheduler, best_loss)

            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch + 1, optimizer, scheduler, best_loss)

        logger.info("Training complete!")

        # Save final model
        self.save_checkpoint('final', optimizer, scheduler, best_loss)

        # Close wandb
        if self.config['logging']['use_wandb']:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train LLM with NDML')
    parser.add_argument('--config', required=True, help='Training configuration file')
    args = parser.parse_args()

    # Create trainer and run training
    trainer = NDMLTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()